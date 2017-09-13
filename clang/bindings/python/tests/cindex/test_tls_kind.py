
from clang.cindex import TLSKind
from clang.cindex import Cursor
from clang.cindex import TranslationUnit

from .util import get_cursor
from .util import get_tu

def test_tls_kind():
    """Ensure that thread-local storage kinds are available on cursors."""

    tu = get_tu("""
int tls_none;
thread_local int tls_dynamic;
_Thread_local int tls_static;
""", lang = 'cpp')

    tls_none = get_cursor(tu.cursor, 'tls_none')
    assert tls_none.tls_kind == TLSKind.NONE;

    tls_dynamic = get_cursor(tu.cursor, 'tls_dynamic')
    assert tls_dynamic.tls_kind == TLSKind.DYNAMIC

    tls_static = get_cursor(tu.cursor, 'tls_static')
    assert tls_static.tls_kind == TLSKind.STATIC

    # The following case tests '__declspec(thread)'.  Since it is a Microsoft
    # specific extension, specific flags are required for the parser to pick
    # these up.
    flags = ['-fms-extensions', '-target', 'x86_64-unknown-windows-win32']
    tu = get_tu("""
__declspec(thread) int tls_declspec;
""", lang = 'cpp', flags=flags)

    tls_declspec = get_cursor(tu.cursor, 'tls_declspec')
    assert tls_declspec.tls_kind == TLSKind.STATIC

