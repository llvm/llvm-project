#!/usr/bin/env python3
"""
Document Processor for LAT5150DRVMIL RAG System
Processes markdown and text files from documentation directory
"""

import os
import re
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentChunker:
    """Chunks documents into overlapping segments for RAG"""

    def __init__(self, chunk_size=256, overlap=20):
        """
        Args:
            chunk_size: Number of words per chunk (optimal from research: 256)
            overlap: Number of overlapping words (optimal from research: 20)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Chunk text into overlapping segments

        Args:
            text: Input text to chunk
            metadata: Metadata about the source document

        Returns:
            List of chunk dictionaries with text and metadata
        """
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)

            chunks.append({
                'text': chunk_text,
                'metadata': {
                    **metadata,
                    'chunk_index': len(chunks),
                    'start_word': i,
                    'end_word': i + len(chunk_words)
                }
            })

            if i + self.chunk_size >= len(words):
                break

        return chunks


class DocumentProcessor:
    """Processes documentation files into searchable chunks"""

    def __init__(self, docs_directory='00-documentation', use_pdf_cache=True):
        self.docs_dir = Path(docs_directory)
        self.chunker = DocumentChunker(chunk_size=256, overlap=20)
        self.documents = []
        self.chunks = []
        self.use_pdf_cache = use_pdf_cache
        self.pdf_cache_file = Path('rag_system/pdf_cache.json')
        self.pdf_cache = self._load_pdf_cache()

    def _load_pdf_cache(self) -> Dict:
        """Load PDF processing cache"""
        if not self.use_pdf_cache or not self.pdf_cache_file.exists():
            return {}

        try:
            with open(self.pdf_cache_file, 'r') as f:
                return json.load(f)
        except:
            return {}

    def _save_pdf_cache(self):
        """Save PDF processing cache"""
        if not self.use_pdf_cache:
            return

        os.makedirs(self.pdf_cache_file.parent, exist_ok=True)
        with open(self.pdf_cache_file, 'w') as f:
            json.dump(self.pdf_cache, f, indent=2)

    @staticmethod
    def _compute_file_hash(filepath: Path) -> str:
        """Compute MD5 hash of file for change detection"""
        md5 = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        return md5.hexdigest()

    def load_markdown(self, filepath: Path) -> str:
        """Load and clean markdown file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove code blocks for cleaner text (but keep the content)
        content = re.sub(r'```[\s\S]*?```', '', content)

        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r' +', ' ', content)

        return content.strip()

    def _assess_extraction_quality(self, text: str, filepath: Path) -> float:
        """
        Assess quality of PDF text extraction

        Returns:
            Quality score 0.0-1.0 (1.0 = perfect, <0.7 = poor, triggers Donut fallback)
        """
        if not text or len(text.strip()) < 50:
            return 0.0

        # Heuristics for extraction quality
        score = 1.0

        # Check for excessive garbage characters
        total_chars = len(text)
        garbage_chars = sum(1 for c in text if ord(c) > 127 and c not in 'áéíóúñü€£¥©®™°')
        if total_chars > 0:
            garbage_ratio = garbage_chars / total_chars
            if garbage_ratio > 0.3:
                score -= 0.4  # Very poor encoding
            elif garbage_ratio > 0.1:
                score -= 0.2  # Some encoding issues

        # Check for reasonable word density
        words = text.split()
        if len(words) < 10:
            score -= 0.3  # Very few words extracted

        # Check average word length (should be 4-8 chars typically)
        if words:
            avg_word_len = sum(len(w) for w in words) / len(words)
            if avg_word_len < 2 or avg_word_len > 15:
                score -= 0.2  # Unusual word lengths suggest poor extraction

        # Check for excessive punctuation (sign of extraction issues)
        punct_ratio = sum(1 for c in text if c in '.,;:!?') / max(total_chars, 1)
        if punct_ratio > 0.15:
            score -= 0.1

        # Check for repeated characters (common in bad extractions)
        repeated_pattern = re.search(r'(.)\1{5,}', text)
        if repeated_pattern:
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _extract_with_donut(self, filepath: Path) -> str:
        """
        Extract PDF content using Donut vision transformer (slow, high quality)

        Args:
            filepath: Path to PDF file

        Returns:
            Extracted text or empty string on failure
        """
        try:
            from rag_system.donut_pdf_processor import DonutPDFProcessor

            logger.info(f"Using Donut OCR-free extraction for {filepath.name}")

            processor = DonutPDFProcessor(device='cpu', use_quantization=True)
            results = processor.process_pdf(filepath, dpi=150, max_pages=None)

            # Combine all page text
            page_texts = []
            for page in results.get('pages', []):
                if 'text' in page:
                    page_texts.append(f"[Page {page['page_number']}]\n{page['text']}")

            if page_texts:
                return '\n\n'.join(page_texts).strip()
            else:
                logger.warning(f"Donut extraction returned no text for {filepath.name}")
                return ""

        except Exception as e:
            logger.error(f"Donut extraction failed for {filepath.name}: {e}")
            return ""

    def _extract_with_pdfplumber(self, filepath: Path) -> str:
        """
        Extract PDF content using pdfplumber (fast, good for most PDFs)

        Args:
            filepath: Path to PDF file

        Returns:
            Extracted text or empty string on failure
        """
        # Try pdfplumber first
        try:
            import pdfplumber
        except ImportError:
            logger.warning(f"pdfplumber not installed, falling back to PyPDF2 for {filepath}")
            try:
                import PyPDF2
                text_content = []
                with open(filepath, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        if text:
                            text_content.append(text)
                return '\n\n'.join(text_content).strip()
            except Exception as e:
                logger.error(f"PyPDF2 extraction failed for {filepath.name}: {e}")
                return ""

        all_content = []

        try:
            with pdfplumber.open(filepath) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_content = []

                    # Extract text
                    text = page.extract_text()
                    if text:
                        page_content.append(text)

                    # Extract tables as markdown
                    tables = page.extract_tables()
                    if tables:
                        for table_num, table in enumerate(tables, 1):
                            # Convert table to markdown format
                            table_md = f"\n[Table {table_num} on page {page_num}]\n"
                            if table:
                                # Get headers (first row)
                                headers = table[0] if table else []
                                # Create markdown table
                                if headers:
                                    table_md += "| " + " | ".join(str(cell or "") for cell in headers) + " |\n"
                                    table_md += "|" + "|".join(["---" for _ in headers]) + "|\n"

                                # Add data rows
                                for row in table[1:]:
                                    table_md += "| " + " | ".join(str(cell or "") for cell in row) + " |\n"

                            page_content.append(table_md)

                    # Extract images (add metadata)
                    images = page.images
                    if images:
                        page_content.append(f"\n[Page {page_num} contains {len(images)} image(s)]")

                    # Join page content
                    if page_content:
                        all_content.append('\n'.join(page_content))

        except Exception as e:
            logger.error(f"pdfplumber extraction failed for {filepath.name}: {e}")
            # Fallback to basic extraction
            try:
                import PyPDF2
                text_content = []
                with open(filepath, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text:
                            text_content.append(text)
                return '\n\n'.join(text_content).strip()
            except Exception as e2:
                logger.error(f"PyPDF2 fallback also failed for {filepath.name}: {e2}")
                return ""

        # Join all pages
        content = '\n\n'.join(all_content)

        # Clean up excessive whitespace
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r' +', ' ', content)

        return content.strip()

    def load_pdf(self, filepath: Path) -> str:
        """
        Load and extract text, tables, and image descriptions from PDF

        Strategy:
        1. Check cache - if file unchanged, return cached result (100x faster)
        2. Try pdfplumber first (fast, good for 80-90% of PDFs)
        3. Assess extraction quality with heuristics
        4. If quality < threshold, automatically retry with Donut (slow but accurate)
        5. Cache result for next time

        Environment Variables:
        - USE_DONUT_PDF=true: Force Donut for all PDFs (slow)
        - DONUT_AUTO_FALLBACK=true: Enable automatic quality-based fallback (default)
        - DONUT_QUALITY_THRESHOLD=0.7: Quality score threshold (default: 0.7)
        - PDF_CACHE=true: Enable PDF caching (default: True)
        """
        # Check cache first
        if self.use_pdf_cache:
            file_hash = self._compute_file_hash(filepath)
            cache_key = str(filepath)

            if cache_key in self.pdf_cache:
                cached = self.pdf_cache[cache_key]
                if cached.get('hash') == file_hash:
                    logger.info(f"✓ Using cached extraction for {filepath.name} (method: {cached.get('method', 'unknown')})")
                    return cached.get('content', '')

        # Check environment variables
        force_donut = os.getenv('USE_DONUT_PDF', 'false').lower() == 'true'
        enable_auto_fallback = os.getenv('DONUT_AUTO_FALLBACK', 'true').lower() == 'true'
        quality_threshold = float(os.getenv('DONUT_QUALITY_THRESHOLD', '0.7'))

        extraction_method = 'pdfplumber'
        quality_score = 0.0

        # Force Donut mode (use for all PDFs - slow but highest quality)
        if force_donut:
            logger.info(f"Force Donut mode enabled for {filepath.name}")
            result_text = self._extract_with_donut(filepath)
            extraction_method = 'donut'
        else:
            # Try pdfplumber first (fast path)
            pdfplumber_text = self._extract_with_pdfplumber(filepath)

            # Assess quality and potentially fallback to Donut
            if enable_auto_fallback and pdfplumber_text:
                quality_score = self._assess_extraction_quality(pdfplumber_text, filepath)

                if quality_score < quality_threshold:
                    logger.warning(
                        f"Low quality extraction (score: {quality_score:.2f}) for {filepath.name}, "
                        f"trying Donut fallback..."
                    )
                    donut_text = self._extract_with_donut(filepath)

                    if donut_text:
                        donut_quality = self._assess_extraction_quality(donut_text, filepath)
                        if donut_quality > quality_score:
                            logger.info(
                                f"Donut improved quality for {filepath.name}: "
                                f"{quality_score:.2f} → {donut_quality:.2f}"
                            )
                            result_text = donut_text
                            extraction_method = 'donut'
                            quality_score = donut_quality
                        else:
                            logger.info(
                                f"Donut quality not better ({donut_quality:.2f}), "
                                f"keeping pdfplumber result"
                            )
                            result_text = pdfplumber_text
                else:
                    logger.info(f"Good quality extraction (score: {quality_score:.2f}) for {filepath.name}")
                    result_text = pdfplumber_text
            else:
                result_text = pdfplumber_text

        # Cache the result
        if self.use_pdf_cache:
            self.pdf_cache[cache_key] = {
                'hash': file_hash,
                'content': result_text,
                'method': extraction_method,
                'quality_score': quality_score,
                'filepath': str(filepath)
            }
            self._save_pdf_cache()

        return result_text

    def process_all_documents(self) -> List[Dict]:
        """
        Process all markdown files in documentation directory

        Returns:
            List of document chunks with metadata
        """
        all_chunks = []

        # Find all documentation files
        md_files = list(self.docs_dir.rglob('*.md'))
        txt_files = list(self.docs_dir.rglob('*.txt'))
        pdf_files = list(self.docs_dir.rglob('*.pdf'))

        all_files = md_files + txt_files + pdf_files

        print(f"Found {len(all_files)} documentation files ({len(md_files)} .md, {len(txt_files)} .txt, {len(pdf_files)} .pdf)")

        for filepath in all_files:
            try:
                # Load content based on file type
                if filepath.suffix.lower() == '.pdf':
                    content = self.load_pdf(filepath)
                else:
                    content = self.load_markdown(filepath)

                if len(content) < 50:  # Skip very short files
                    continue

                metadata = {
                    'filename': filepath.name,
                    'filepath': str(filepath.relative_to(self.docs_dir)),
                    'file_size': len(content),
                    'category': filepath.parent.name
                }

                chunks = self.chunker.chunk_text(content, metadata)
                all_chunks.extend(chunks)

                self.documents.append({
                    'filepath': filepath,
                    'content': content,
                    'metadata': metadata,
                    'num_chunks': len(chunks)
                })

            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue

        self.chunks = all_chunks
        print(f"Created {len(all_chunks)} chunks from {len(self.documents)} documents")

        return all_chunks

    def save_processed_data(self, output_path='rag_system/processed_docs.json'):
        """Save processed chunks to JSON for quick loading"""
        data = {
            'chunks': self.chunks,
            'documents': [
                {
                    'filepath': str(doc['filepath']),
                    'metadata': doc['metadata'],
                    'num_chunks': doc['num_chunks']
                }
                for doc in self.documents
            ],
            'stats': {
                'total_documents': len(self.documents),
                'total_chunks': len(self.chunks),
                'avg_chunks_per_doc': len(self.chunks) / len(self.documents) if self.documents else 0
            }
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print(f"Saved processed data to {output_path}")
        return output_path


if __name__ == '__main__':
    # Process documents and build embeddings
    processor = DocumentProcessor('00-documentation')
    chunks = processor.process_all_documents()
    processor.save_processed_data()

    print("\n" + "="*60)
    print("Document Processing Complete")
    print("="*60)
    print(f"Total chunks: {len(chunks)}")
    print(f"Output: rag_system/processed_docs.json")
    print("\nNext: Run transformer_upgrade.py to build embeddings")
    print("="*60)
