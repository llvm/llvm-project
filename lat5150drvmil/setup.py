"""
LAT5150DRVMIL Setup Configuration
Python package setup for submodule integration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text().splitlines()
        if line.strip() and not line.startswith('#')
    ]

setup(
    name="LAT5150DRVMIL",
    version="1.0.0",
    author="LAT5150DRVMIL Project",
    author_email="",
    description="Dell Latitude 5450 Covert AI Platform - Military-Grade Local AI System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SWORDIntel/LAT5150DRVMIL",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        'screenshot_intel': [
            'qdrant-client>=1.7.0',
            'sentence-transformers>=2.2.0',
            'paddleocr>=2.7.0',
            'paddlepaddle>=2.5.0',
            'pytesseract>=0.3.10',
            'Pillow>=10.0.0',
            'psutil>=5.9.0',
        ],
        'telegram': [
            'telethon>=1.34.0',
        ],
        'api': [
            'fastapi>=0.104.0',
            'uvicorn[standard]>=0.24.0',
            'pydantic>=2.0.0',
        ],
        'dev': [
            'pytest>=7.4.0',
            'pytest-asyncio>=0.21.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'lat5150-validate=LAT5150DRVMIL.ai_engine.system_validator:main',
            'lat5150-screenshot-intel=LAT5150DRVMIL.screenshot_intel.screenshot_intel_cli:main',
        ],
    },
    include_package_data=True,
    package_data={
        'LAT5150DRVMIL': [
            '02-ai-engine/*.json',
            '06-intel-systems/**/*.md',
            '00-documentation/**/*.md',
        ],
    },
    zip_safe=False,
)
