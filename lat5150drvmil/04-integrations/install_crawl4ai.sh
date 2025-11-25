#!/bin/bash
# Install crawl4ai and all dependencies

echo "Installing crawl4ai dependencies..."

# Core dependencies
pip3 install -q aiosqlite aiofiles aiohttp anyio lxml \
    numpy pillow playwright patchright python-dotenv \
    requests beautifulsoup4 xxhash rank-bm25 \
    snowballstemmer pydantic pyOpenSSL psutil \
    tf-playwright-stealth litellm

# Optional but recommended
pip3 install -q PyPDF2 torch nltk scikit-learn

# Install playwright browsers
playwright install chromium

echo "✅ crawl4ai dependencies installed"
echo ""
echo "Test with:"
echo "  python3 /home/john/LAT5150DRVMIL/04-integrations/crawl4ai_wrapper.py https://example.com"
