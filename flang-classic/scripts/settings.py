from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
FLANG_DIR = THIS_DIR.parent
TOOLCHAIN_DIR = THIS_DIR.joinpath('cmake')
LIBPGMATH_DIR = FLANG_DIR.joinpath('runtime', 'libpgmath')
