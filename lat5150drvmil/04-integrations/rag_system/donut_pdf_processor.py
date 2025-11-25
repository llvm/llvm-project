#!/usr/bin/env python3
"""
Donut PDF Processor - OCR-free document understanding
Uses Naver Clova's Donut model for visual document extraction

Donut (Document Understanding Transformer):
- Vision Encoder: Swin Transformer for image embedding
- Text Decoder: BART for autoregressive text generation
- OCR-free: Direct image-to-text processing
- Best for: Structured documents, forms, receipts, scientific papers with complex layouts

References:
- Model: https://huggingface.co/naver-clova-ix/donut-base
- Paper: https://arxiv.org/abs/2111.15664
- GitHub: https://github.com/clovaai/donut
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    from PIL import Image
    import torch
    DONUT_AVAILABLE = True
except ImportError:
    logger.warning("Donut dependencies not available. Install with: pip install transformers torch pillow")
    DONUT_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    logger.warning("pdf2image not available. Install with: pip install pdf2image")
    logger.warning("Also requires poppler-utils: sudo apt-get install poppler-utils")
    PDF2IMAGE_AVAILABLE = False


class DonutPDFProcessor:
    """
    Donut-based PDF processor for OCR-free document understanding

    Features:
    - Converts PDF pages to images
    - Processes with Donut vision-text model
    - Extracts structured content without OCR
    - Handles complex layouts, forms, and scientific papers
    """

    def __init__(
        self,
        model_name: str = "naver-clova-ix/donut-base",
        device: str = "cpu",
        use_quantization: bool = True
    ):
        """
        Initialize Donut PDF processor

        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on (cpu/cuda)
            use_quantization: Apply INT8 quantization for faster CPU inference
        """
        if not DONUT_AVAILABLE:
            raise ImportError(
                "Donut dependencies not installed!\n"
                "Install with: pip install transformers torch pillow"
            )

        if not PDF2IMAGE_AVAILABLE:
            raise ImportError(
                "pdf2image not installed!\n"
                "Install with: pip install pdf2image\n"
                "Also requires: sudo apt-get install poppler-utils"
            )

        logger.info(f"Initializing Donut processor: {model_name}")

        self.device = device
        self.model_name = model_name

        # Load processor and model
        self.processor = DonutProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

        # Move to device
        self.model = self.model.to(device)
        self.model.eval()

        # Apply quantization for CPU
        if use_quantization and device == "cpu":
            try:
                from optimum.quanto import quantize, freeze, qint8
                logger.info("Applying INT8 quantization for faster CPU inference...")
                quantize(self.model, weights=qint8)
                freeze(self.model)
                logger.info("✓ Model quantized to INT8")
            except ImportError:
                logger.warning("optimum-quanto not available, skipping quantization")
            except Exception as e:
                logger.warning(f"Quantization failed: {e}")

        logger.info("Donut processor initialized")

    def pdf_to_images(
        self,
        pdf_path: Path,
        dpi: int = 200,
        max_pages: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Convert PDF pages to images

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for rendering (higher = better quality but slower)
            max_pages: Maximum number of pages to process (None = all)

        Returns:
            List of PIL Images (one per page)
        """
        logger.info(f"Converting PDF to images: {pdf_path}")

        try:
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=1,
                last_page=max_pages
            )
            logger.info(f"Converted {len(images)} pages to images")
            return images
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            return []

    def process_image(
        self,
        image: Image.Image,
        task_prompt: str = "<s>",
        max_length: int = 512
    ) -> str:
        """
        Process a single image with Donut

        Args:
            image: PIL Image of document page
            task_prompt: Task-specific prompt (e.g., "<s_cord-v2>" for receipts)
            max_length: Maximum tokens to generate

        Returns:
            Extracted text from image
        """
        # Prepare inputs
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        # Prepare decoder input
        decoder_input_ids = self.processor.tokenizer(
            task_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.to(self.device)

        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=max_length,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        # Decode output
        sequence = self.processor.batch_decode(outputs.sequences)[0]

        # Remove special tokens
        sequence = sequence.replace(task_prompt, "").strip()
        sequence = self.processor.token2json(sequence) if hasattr(self.processor, 'token2json') else sequence

        return sequence

    def process_pdf(
        self,
        pdf_path: Path,
        dpi: int = 200,
        max_pages: Optional[int] = None,
        task_prompt: str = "<s>"
    ) -> Dict:
        """
        Process entire PDF with Donut

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for rendering
            max_pages: Maximum pages to process
            task_prompt: Task-specific prompt

        Returns:
            Dict with extracted content per page
        """
        logger.info(f"Processing PDF with Donut: {pdf_path}")

        # Convert PDF to images
        images = self.pdf_to_images(pdf_path, dpi=dpi, max_pages=max_pages)

        if not images:
            return {"error": "Failed to convert PDF to images", "pages": []}

        # Process each page
        results = {
            "filename": pdf_path.name,
            "num_pages": len(images),
            "pages": []
        }

        for page_num, image in enumerate(images, 1):
            logger.info(f"Processing page {page_num}/{len(images)}...")

            try:
                text = self.process_image(image, task_prompt=task_prompt)
                results["pages"].append({
                    "page_number": page_num,
                    "text": text,
                    "image_size": image.size
                })
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}")
                results["pages"].append({
                    "page_number": page_num,
                    "error": str(e)
                })

        logger.info(f"✓ Processed {len(images)} pages from {pdf_path.name}")
        return results

    def get_model_info(self) -> Dict:
        """Get information about loaded model"""
        param_count = sum(p.numel() for p in self.model.parameters())
        param_size_mb = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        ) / (1024 ** 2)

        return {
            'model_name': self.model_name,
            'device': self.device,
            'parameters': f"{param_count:,}",
            'model_size_mb': f"{param_size_mb:.1f} MB",
            'vision_encoder': 'Swin Transformer',
            'text_decoder': 'BART',
            'ocr_free': True
        }


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Donut PDF Processor')
    parser.add_argument('pdf_file', help='PDF file to process')
    parser.add_argument('--dpi', type=int, default=200, help='DPI for rendering')
    parser.add_argument('--max-pages', type=int, help='Maximum pages to process')
    parser.add_argument('--model', default='naver-clova-ix/donut-base', help='Donut model to use')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--no-quantize', action='store_true', help='Disable quantization')

    args = parser.parse_args()

    # Initialize processor
    processor = DonutPDFProcessor(
        model_name=args.model,
        device=args.device,
        use_quantization=not args.no_quantize
    )

    # Show model info
    info = processor.get_model_info()
    print("\n" + "=" * 70)
    print("Donut Model Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print("=" * 70 + "\n")

    # Process PDF
    results = processor.process_pdf(
        Path(args.pdf_file),
        dpi=args.dpi,
        max_pages=args.max_pages
    )

    # Display results
    print(f"\nProcessed: {results['filename']}")
    print(f"Pages: {results['num_pages']}")
    print("=" * 70)

    for page in results['pages']:
        print(f"\nPage {page['page_number']}:")
        print("-" * 70)
        if 'error' in page:
            print(f"Error: {page['error']}")
        else:
            print(page['text'])
        print()


if __name__ == '__main__':
    main()
