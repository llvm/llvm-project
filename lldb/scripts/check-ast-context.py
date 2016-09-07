#!/usr/bin/env python

import sys
import subprocess
import os
import os.path
import argparse
import json
import hashlib


class Hasher(object):

    @classmethod
    def from_file(cls, file):
        hl = hashlib.md5()
        hl.update(file.read())
        return hl.hexdigest()


class IncludePath(object):

    def __init__(self, path):
        self.path = path

    def to_arg(self):
        return ['-I', self.path]


class Macro(object):

    def __init__(self, macro):
        self.macro = macro

    def to_arg(self):
        return ['-D%s' % self.macro]


class SDKRoot(object):

    def __init__(self, root):
        self.root = root

    def to_arg(self):
        return ['-isysroot', self.root]


class CPP11(object):

    def __init__(self):
        pass

    def to_arg(self):
        return ['-x', 'c++', '-std=c++11']


class Parser(object):

    def __init__(self, parser):
        self.parser = parser
        self.cursor = self.parser.cursor

    def find_if_impl(self, cursor, f):
        results = []
        if f(cursor):
            results.append(cursor)
        for c in cursor.get_children():
            results += self.find_if_impl(c, f)
        return results

    def find_if(self, f):
        return self.find_if_impl(self.cursor, f)


class Index(object):

    def __init__(self):
        self.index = clang.cindex.Index(
            clang.cindex.conf.lib.clang_createIndex(
                False, True))

    def parse(self, file, options):
        opts = []
        for opt in options:
            opts += opt.to_arg()
        return Parser(self.index.parse(file, opts))


def get_args():
    parser = argparse.ArgumentParser(
        description="Ensure that LLDB's SwiftASTContext class safely uses the underlying swift::ASTContext object")
    parser.add_argument(
        '--file',
        type=str,
        help='path to SwiftASTContext.cpp',
        required=True)
    parser.add_argument(
        '--llvmbuild',
        type=str,
        help='location of the LLVM build tree',
        required=True)
    parser.add_argument(
        '--llvmbarch',
        type=str,
        help='LLVM build directory architecture',
        default=None)
    parser.add_argument(
        '--lldbbuild',
        type=str,
        help='location of the LLDB build tree',
        required=True)
    parser.add_argument(
        '--swiftbuild',
        type=str,
        help='location of the Swift build tree',
        required=True)
    parser.add_argument(
        '--sdk',
        type=str,
        help='location of the SDK root',
        default=None)
    parser.add_argument('--verbose', type=bool, help='verbose output')

    return parser.parse_args(sys.argv[1:])


def detect_source_layout(args):
    args.lldb = os.path.abspath(
        os.path.join(
            os.path.dirname(
                args.file),
            '..',
            '..'))
    args.header = os.path.join(
        args.lldb,
        'include',
        'lldb',
        'Symbol',
        'SwiftASTContext.h')
    if not(
        os.path.exists(
            os.path.join(
            args.llvmbuild,
            'lib',
            'libclang.dylib'))):
        if os.path.exists(
            os.path.join(
                args.llvmbuild,
                args.llvmbarch,
                'lib',
                'libclang.dylib')):
            args.llvmbuild = os.path.join(args.llvmbuild, args.llvmbarch)
    if os.path.isdir(
        os.path.join(
            args.lldb,
            'llvm')) and os.path.isdir(
            os.path.join(
                args.lldb,
                'llvm',
                'tools',
                'clang')) and os.path.isdir(
                    os.path.join(
                        args.lldb,
                        'llvm',
                        'tools',
                        'swift')):
        args.source = args.lldb
        args.llvm = os.path.join(args.source, 'llvm')
        args.clang = os.path.join(args.source, 'llvm', 'tools', 'clang')
        args.swift = os.path.join(args.source, 'llvm', 'tools', 'swift')
        return True
    args.parent = os.path.abspath(os.path.join(args.lldb, '..'))
    if os.path.isdir(os.path.join(args.parent, 'lldb')) and \
       os.path.isdir(os.path.join(args.parent, 'swift')) and \
       os.path.isdir(os.path.join(args.parent, 'clang')) and \
       os.path.isdir(os.path.join(args.parent, 'llvm')):
        args.source = args.parent
        args.swift = os.path.join(args.source, 'swift')
        args.clang = os.path.join(args.source, 'clang')
        args.llvm = os.path.join(args.source, 'llvm')
        return True
    if args.verbose:
        print('arg dictionary = %s' % args)
    return False


def makehashes(args):
    hashes = {}
    with open(args.file) as f:
        hashes['cpp'] = Hasher.from_file(f)
    with open(args.header) as f:
        hashes['h'] = Hasher.from_file(f)
    return hashes


def readhashes(args):
    try:
        p = os.path.join(args.lldbbuild, 'check-ast-context.md5')
        if os.path.exists(p):
            with open(p, 'r') as f:
                return json.load(f)
    finally:
        pass
    return None


def comparehashes(args):
    made = makehashes(args)
    read = readhashes(args)
    if read is None:
        return False
    made_cpp = made.get('cpp')
    read_cpp = read.get('cpp')
    if made_cpp is None or read_cpp is None:
        return False
    if made_cpp != read_cpp:
        return False
    made_h = made.get('h')
    read_h = read.get('h')
    if made_h is None or read_h is None:
        return False
    if made_h != read_h:
        return False
    return True


def writehashes(args):
    try:
        hashes = makehashes(args)
        p = os.path.join(args.lldbbuild, 'check-ast-context.md5')
        with open(p, 'w') as f:
            json.dump(hashes, f)
    finally:
        pass


def init_libclang(src_path, lib_path):
    def look_for_node(cursor, f):
        if not cursor:
            return None
        if not f:
            return None
        if f(cursor):
            return cursor
        for c in cursor.get_children():
            w = look_for_node(c, f)
            if w:
                return w
        return None

    def printtree(node, depth=0):
        d = '  ' * depth
        print('%s%s (%s)') % (d, node.pretty_print(), node.kind)
        for c in node.get_children():
            printtree(c, depth + 1)

    try:
        globals()['clang'] = __import__('clang')
        globals()['clang.cindex'] = __import__('clang.cindex')
    except:
        sys.path.insert(1, src_path)
        try:
            globals()['clang'] = __import__('clang')
            globals()['clang.cindex'] = __import__('clang.cindex')
        except:
            return False
    clang.cindex.Config.set_library_path(lib_path)
    clang.cindex.Cursor.pretty_print = lambda self: (
        self.spelling or self.displayname or self.mangled_name)
    clang.cindex.Cursor.search = look_for_node
    clang.cindex.Cursor.dump = lambda self: printtree(self, depth=0)
    return True


def main():
    args = get_args()
    if not args.sdk:
        args.sdk = subprocess.check_output(
            'xcrun --sdk macosx --show-sdk-path', shell=True).strip()
    detect_source_layout(args)
    if comparehashes(args):
        if args.verbose:
            print('MD5 matches; skipping check')
        return 0
    src_path = os.path.join(args.clang, 'bindings', 'python')
    lib_path = os.path.join(args.llvmbuild, 'lib')
    if not init_libclang(src_path, lib_path):
        print('libclang initialization failed - please try again')

    index = Index()

    macros = [
        Macro(x) for x in [
            '__STDC_CONSTANT_MACROS',
            '__STDC_LIMIT_MACROS']]
    includes = [IncludePath(x) for x in [
        os.path.join(os.path.abspath(args.llvm), 'include'),
        os.path.join(os.path.abspath(args.clang), 'include'),
        os.path.join(os.path.abspath(args.swift), 'include'),
        os.path.join(os.path.abspath(args.lldb), 'include'),
        os.path.join(os.path.abspath(args.lldb), 'source'),
        os.path.join(os.path.abspath(args.llvmbuild), 'include'),
        os.path.join(os.path.abspath(args.llvmbuild), 'tools', 'clang', 'include'),
        os.path.join(os.path.abspath(args.swiftbuild), 'include'),
    ]]
    lang = [CPP11()]
    sdk = [SDKRoot(args.sdk)]
    parser = index.parse(
        args.file,
        macros + lang + sdk + includes)

    def search_lambda(cursor):
        try:
            if not (cursor.kind == clang.cindex.CursorKind.CXX_METHOD):
                return False
            parent = cursor.semantic_parent
            if not ((parent.spelling or parent.displayname)
                    == 'SwiftASTContext'):
                return False
            if not (cursor.is_definition()):
                return False
            if (cursor.is_static_method()):
                return False
            return True
        except:
            return False

    methods = parser.find_if(search_lambda)

    FAIL = 0

    def emit_fail(method):
        print(
            '%s:%s:%s: error: %s not found on method \'%s\'; consider adding' %
            (os.path.basename(
                method.location.file.name),
                method.location.line,
                method.location.column,
                ('VALID_OR_RETURN_VOID' if method.result_type.kind == clang.cindex.TypeKind.VOID else 'VALID_OR_RETURN'),
                method.pretty_print()))

    def scan(method):
        look_for_COMPOUND_STMT = lambda c: c.kind == clang.cindex.CursorKind.COMPOUND_STMT
        look_for_DO_STMT = lambda c: c.kind == clang.cindex.CursorKind.DO_STMT
        look_for_CALL_EXPR = lambda c: c.kind == clang.cindex.CursorKind.CALL_EXPR
        look_for_MEMBER_REF_EXPR = lambda c: c.kind == clang.cindex.CursorKind.MEMBER_REF_EXPR
        compound_stmt = method.search(look_for_COMPOUND_STMT)
        if not compound_stmt:
            return False
        do_stmt = compound_stmt.search(look_for_DO_STMT)
        if not do_stmt:
            return False
        call_expr = do_stmt.search(look_for_CALL_EXPR)
        if not call_expr:
            return False
        member_ref_expr = call_expr.search(look_for_MEMBER_REF_EXPR)
        if not member_ref_expr:
            return False
        if member_ref_expr.pretty_print() == 'HasFatalErrors':
            return True
        return False

    def is_safe(method):
        def look_for_IVAR(c):
            if c.kind == clang.cindex.CursorKind.MEMBER_REF_EXPR:
                if c.pretty_print() == 'm_ast_context_ap':
                    return True
            return False

        def look_for_METHOD(c):
            if c.kind == clang.cindex.CursorKind.CALL_EXPR:
                if c.pretty_print() == 'GetASTContext':
                    return True
            return False
        unsafe = method.search(
            lambda c: look_for_IVAR(c) or look_for_METHOD(c))
        return (unsafe is None)

    whitelist = [
        'GetPluginName',
        'GetPluginVersion',
        'HasFatalErrors',
        'GetFatalErrors'
    ]

    for method in methods:
        if method.pretty_print() in whitelist:
            continue
        try:
            if not scan(method) and not is_safe(method):
                emit_fail(method)
                FAIL += 1
        except Exception as e:
            print(e)
            emit_fail(method)
            FAIL += 1

    if FAIL == 0:
        writehashes(args)
    return FAIL

if main() > 0:
    sys.exit(1)
