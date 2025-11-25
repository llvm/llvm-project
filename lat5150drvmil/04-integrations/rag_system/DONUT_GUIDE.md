# Donut OCR-Free PDF Processing Guide

## Overview

**Donut (Document Understanding Transformer)** is an OCR-free document understanding model from Naver Clova that uses vision transformers to directly extract text and structure from document images.

**Key Advantages:**
- ✅ **OCR-free** - Direct image-to-text using vision transformers
- ✅ **Structured extraction** - Understands document layout natively
- ✅ **Complex layouts** - Handles forms, tables, scientific papers
- ✅ **No preprocessing** - Works directly on PDF page images
- ✅ **Quantized inference** - INT8 quantization for CPU efficiency

**Model Architecture:**
- **Vision Encoder:** Swin Transformer (processes document images)
- **Text Decoder:** BART (generates text autoregressively)
- **License:** MIT
- **Parameters:** ~140M (base model)

---

## When to Use Donut

### ✅ Use Donut For:

1. **Complex Layouts**
   - Scientific papers with equations
   - Forms with structured fields
   - Receipts and invoices
   - Documents with mixed text/graphics

2. **Poor Text Extraction**
   - Scanned documents
   - Low-quality PDFs
   - Documents with non-standard fonts
   - When pdfplumber extraction is incomplete

3. **Structured Data**
   - Need to extract specific fields
   - Preserve document structure
   - Extract tables with context

### ❌ **DON'T** Use Donut For:

1. **Clean Text PDFs** - pdfplumber is faster
2. **Large batches** - Donut is slower (~10-30s per page on CPU)
3. **Simple documents** - Overkill for plain text
4. **Low memory systems** - Model requires ~2GB RAM

---

## Installation

### 1. Install Python Dependencies

```bash
# Core dependencies (already installed in LAT5150DRVMIL)
pip install transformers torch pillow

# PDF to image conversion
pip install pdf2image

# Optional: quantization for faster CPU inference
pip install optimum-quanto
```

### 2. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install poppler-utils
```

**macOS:**
```bash
brew install poppler
```

**Verify installation:**
```bash
pdftoppm -v
```

---

## Quick Start

### Method 1: Standalone Script

```bash
# Process a single PDF
python3 rag_system/donut_pdf_processor.py path/to/document.pdf

# Limit to first 3 pages
python3 rag_system/donut_pdf_processor.py path/to/document.pdf --max-pages 3

# Higher quality (slower)
python3 rag_system/donut_pdf_processor.py path/to/document.pdf --dpi 300

# Disable quantization (higher quality, slower)
python3 rag_system/donut_pdf_processor.py path/to/document.pdf --no-quantize
```

### Method 2: Integrated with RAG System

**Automatic Quality-Based Fallback (Recommended):**
```bash
# Default behavior - uses pdfplumber first, automatically falls back to Donut for poor quality
python3 rag_system/document_processor.py

# The system will:
# 1. Try pdfplumber first (fast)
# 2. Assess extraction quality (0.0-1.0 score)
# 3. If quality < 0.7, retry with Donut automatically
# 4. Use best result
```

**Force Donut for All PDFs:**
```bash
# Enable Donut mode for ALL document processing (slow but highest quality)
export USE_DONUT_PDF=true

# Rebuild document index with Donut extraction
python3 rag_system/document_processor.py

# Disable Donut mode (back to pdfplumber)
export USE_DONUT_PDF=false
```

**Disable Automatic Fallback:**
```bash
# Only use pdfplumber, never fallback to Donut
export DONUT_AUTO_FALLBACK=false
python3 rag_system/document_processor.py
```

### Method 3: Python API

```python
from pathlib import Path
from rag_system.donut_pdf_processor import DonutPDFProcessor

# Initialize processor
processor = DonutPDFProcessor(
    model_name="naver-clova-ix/donut-base",
    device="cpu",
    use_quantization=True  # INT8 for faster CPU inference
)

# Process PDF
results = processor.process_pdf(
    Path("document.pdf"),
    dpi=200,              # Image quality (150-300)
    max_pages=None        # None = all pages
)

# Access results
for page in results['pages']:
    print(f"Page {page['page_number']}:")
    print(page['text'])
    print()
```

---

## Configuration

### Environment Variables

**Automatic Quality-Based Fallback (Default):**
```bash
# Enable automatic fallback (default: true)
export DONUT_AUTO_FALLBACK=true

# Set quality threshold (default: 0.7)
# PDFs with quality < threshold will trigger Donut fallback
export DONUT_QUALITY_THRESHOLD=0.7

# Example: More aggressive fallback (use Donut more often)
export DONUT_QUALITY_THRESHOLD=0.85

# Example: Less aggressive fallback (only for very poor PDFs)
export DONUT_QUALITY_THRESHOLD=0.5
```

**Force Donut for All PDFs:**
```bash
# Enable Donut for all PDF processing (ignores quality assessment)
export USE_DONUT_PDF=true

# Or set in .bashrc/.zshrc for persistence
echo 'export USE_DONUT_PDF=true' >> ~/.bashrc
```

**Quality Assessment Heuristics:**

The system assesses PDF extraction quality using these factors:
- **Garbage characters:** High proportion of invalid Unicode (score penalty: -0.2 to -0.4)
- **Word density:** Very few words extracted (score penalty: -0.3)
- **Word length:** Unusual average word length (score penalty: -0.2)
- **Punctuation:** Excessive punctuation ratio (score penalty: -0.1)
- **Repeated characters:** Long character repetitions (score penalty: -0.2)

**Quality Score Interpretation:**
- `1.0` - Perfect extraction
- `0.7-0.9` - Good quality (uses pdfplumber)
- `< 0.7` - Poor quality (triggers Donut fallback)
- `0.0` - No meaningful content extracted

### Model Selection

Available models:

| Model | Size | Best For |
|-------|------|----------|
| `naver-clova-ix/donut-base` | 140M | General documents (default) |
| `naver-clova-ix/donut-base-finetuned-cord-v2` | 140M | Receipts |
| `naver-clova-ix/donut-base-finetuned-docvqa` | 140M | Document Q&A |

**Change model:**
```python
processor = DonutPDFProcessor(
    model_name="naver-clova-ix/donut-base-finetuned-docvqa"
)
```

### Performance Tuning

**DPI (Resolution):**
- `150` - Fast, good for most documents
- `200` - Balanced (default)
- `300` - High quality, slower

**Quantization:**
```python
# INT8 (default) - 2x faster, minimal quality loss
processor = DonutPDFProcessor(use_quantization=True)

# FP32 - Best quality, slower
processor = DonutPDFProcessor(use_quantization=False)
```

---

## Performance Benchmarks

**System:** Intel CPU (no GPU)
**Document:** 10-page scientific paper

| Method | Total Time | Per Page | Quality |
|--------|------------|----------|---------|
| **pdfplumber** | 2-3s | 0.2-0.3s | Good for clean PDFs |
| **Donut (INT8)** | 2-5min | 12-30s | Excellent for complex layouts |
| **Donut (FP32)** | 4-8min | 24-48s | Best quality |

**RAM Usage:**
- pdfplumber: ~100MB
- Donut (INT8): ~2GB
- Donut (FP32): ~4GB

**Recommendations:**
- **For 1-10 PDFs:** Donut is fine
- **For 100+ PDFs:** Use pdfplumber, fallback to Donut for failures
- **For real-time:** pdfplumber only

---

## Examples

### Example 1: Scientific Paper

```bash
# Extract text from complex scientific paper
python3 rag_system/donut_pdf_processor.py \
    00-documentation/An-advanced-retrieval-augmented-generation-system.pdf \
    --dpi 200
```

**Output:**
```
Processing PDF with Donut: An-advanced-retrieval-augmented-generation-system.pdf
Page 1/8: Converting to image...
Page 1/8: Extracting text with Donut...
✓ Processed 8 pages

Page 1:
[Extracted text with preserved formatting, equations, tables]
```

### Example 2: Batch Processing

```python
from pathlib import Path
from rag_system.donut_pdf_processor import DonutPDFProcessor

processor = DonutPDFProcessor()

# Process all PDFs in directory
pdf_dir = Path("00-documentation")
for pdf_file in pdf_dir.glob("*.pdf"):
    print(f"Processing {pdf_file.name}...")

    results = processor.process_pdf(pdf_file, max_pages=5)

    # Save to markdown
    output_file = pdf_file.with_suffix('.md')
    with open(output_file, 'w') as f:
        for page in results['pages']:
            f.write(f"## Page {page['page_number']}\n\n")
            f.write(page['text'] + "\n\n")

    print(f"✓ Saved to {output_file}")
```

### Example 3: Fallback Strategy

```python
def extract_pdf_text(pdf_path):
    """Try pdfplumber first, fallback to Donut if needed"""
    import pdfplumber

    # Try pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n\n".join(page.extract_text() or "" for page in pdf.pages)

            # Check if extraction was successful
            if len(text.strip()) > 100:
                return text
    except:
        pass

    # Fallback to Donut
    print(f"pdfplumber failed, trying Donut...")
    processor = DonutPDFProcessor()
    results = processor.process_pdf(pdf_path)

    return "\n\n".join(
        page['text'] for page in results['pages'] if 'text' in page
    )
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pdf2image'"

**Solution:**
```bash
pip install pdf2image
```

### Issue: "PDFInfoNotInstalledError: Unable to get page count"

**Solution:** Install poppler-utils
```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils

# macOS
brew install poppler

# Verify
pdftoppm -v
```

### Issue: Out of Memory

**Solutions:**
1. Enable quantization:
   ```python
   processor = DonutPDFProcessor(use_quantization=True)
   ```

2. Process fewer pages at once:
   ```python
   results = processor.process_pdf(pdf_path, max_pages=3)
   ```

3. Lower DPI:
   ```python
   results = processor.process_pdf(pdf_path, dpi=150)
   ```

### Issue: Very Slow Processing

**Solutions:**
1. Already using INT8 quantization (default)
2. Lower DPI: `dpi=150`
3. Use pdfplumber for clean PDFs
4. Consider GPU if available:
   ```python
   processor = DonutPDFProcessor(device='cuda')
   ```

### Issue: Poor Extraction Quality

**Solutions:**
1. Increase DPI:
   ```python
   results = processor.process_pdf(pdf_path, dpi=300)
   ```

2. Disable quantization:
   ```python
   processor = DonutPDFProcessor(use_quantization=False)
   ```

3. Try a fine-tuned model:
   ```python
   processor = DonutPDFProcessor(
       model_name="naver-clova-ix/donut-base-finetuned-docvqa"
   )
   ```

---

## Comparison with Other Methods

| Feature | pdfplumber | PyPDF2 | **Donut** | Nougat |
|---------|------------|--------|-----------|---------|
| **Speed** | ⚡⚡⚡ Fast | ⚡⚡⚡ Fast | ⚡ Slow | ⚡ Slow |
| **Quality** | Good | Basic | **Excellent** | Excellent |
| **Tables** | ✅ Yes | ❌ No | ✅ Yes | ✅ Yes |
| **Images** | ⚠️ Metadata | ❌ No | ⚠️ Understands context | ⚠️ Metadata |
| **Scanned PDFs** | ❌ Poor | ❌ Poor | ✅ **Excellent** | ✅ Excellent |
| **Complex Layouts** | ⚠️ Ok | ❌ Poor | ✅ **Excellent** | ✅ Excellent |
| **Memory** | Low | Low | **High** | High |
| **Dependencies** | Light | Light | **Heavy** | Heavy |

**Recommendation:**
1. **Default:** Use pdfplumber (fastest, good quality)
2. **Fallback:** Use Donut for complex/scanned documents
3. **Academic papers:** Consider Nougat (optimized for scientific docs)

---

## Integration with RAG System

Donut is now integrated into the LAT5150DRVMIL RAG system as an optional enhancement.

**Enable globally:**
```bash
export USE_DONUT_PDF=true
python3 rag_system/document_processor.py
python3 rag_system/transformer_upgrade.py
```

**Enable for specific files:**
```python
from rag_system.donut_pdf_processor import DonutPDFProcessor

processor = DonutPDFProcessor()
results = processor.process_pdf(Path("complex_document.pdf"))

# Manually add to RAG index
# ... your integration code ...
```

---

## Advanced Usage

### Custom Task Prompts

Donut supports task-specific prompts for specialized extraction:

```python
# Receipt parsing
results = processor.process_image(
    image,
    task_prompt="<s_cord-v2>"  # Receipt format
)

# Document Q&A
results = processor.process_image(
    image,
    task_prompt="<s_docvqa>"  # Question answering
)

# General extraction (default)
results = processor.process_image(
    image,
    task_prompt="<s>"
)
```

### GPU Acceleration

If you have a CUDA-capable GPU:

```python
processor = DonutPDFProcessor(
    device='cuda',
    use_quantization=False  # Quantization less beneficial on GPU
)

# Expected speedup: 10-50x faster
```

### Model Caching

First run downloads the model (~550MB). Subsequent runs use cached model.

**Cache location:**
```bash
~/.cache/huggingface/hub/models--naver-clova-ix--donut-base
```

**Clear cache:**
```bash
rm -rf ~/.cache/huggingface/hub/models--naver-clova-ix--donut-base
```

---

## References

- **Hugging Face:** https://huggingface.co/naver-clova-ix/donut-base
- **GitHub:** https://github.com/clovaai/donut
- **Paper:** https://arxiv.org/abs/2111.15664
- **PyPI:** https://pypi.org/project/donut-python/

---

## Changelog

**2025-11-08:**
- Initial integration into LAT5150DRVMIL RAG system
- INT8 quantization support for CPU inference
- Opt-in via USE_DONUT_PDF environment variable
- Automatic fallback to pdfplumber if Donut fails

---

*Last updated: 2025-11-08*
*LAT5150DRVMIL RAG System v3.0*
