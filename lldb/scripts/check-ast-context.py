#!/usr/bin/env python

import sys
import subprocess
import os
import os.path
import argparse

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
    return ['-isysroot',self.root]

class CPP11(object):
  def __init__(self):
    pass
  
  def to_arg(self):
    return ['-x','c++','-std=c++11']

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
    self.index = clang.cindex.Index(clang.cindex.conf.lib.clang_createIndex(False, True))

  def parse(self, file, options):
    opts = []
    for opt in options:
      opts += opt.to_arg()
    return Parser(self.index.parse(file, opts))

def get_args():
  parser = argparse.ArgumentParser(description="Ensure that LLDB's SwiftASTContext class safely uses the underlying swift::ASTContext object")
  parser.add_argument('--source',  type=str, help='location of the source tree', required=True)
  parser.add_argument('--build',   type=str, help='location of the build tree', required=True)
  parser.add_argument('--sdk',     type=str, help='location of the SDK root', default=None)
  parser.add_argument('--verbose', type=bool,help='verbose output')

  return parser.parse_args(sys.argv[1:])

def detect_source_layout(args):
  if os.path.isdir(os.path.join(args.source,'lldb')) and \
     os.path.isdir(os.path.join(args.source,'swift')) and \
     os.path.isdir(os.path.join(args.source,'clang')) and \
     os.path.isdir(os.path.join(args.source,'llvm')):
     args.lldb = os.path.join(args.source,'lldb')
     args.swift = os.path.join(args.source,'swift')
     args.clang = os.path.join(args.source,'clang')
     args.llvm = os.path.join(args.source,'llvm')
  else:
    if os.path.isdir(os.path.join(args.source,'llvm')) and \
       os.path.isdir(os.path.join(args.source,'llvm','tools','clang')) and \
       os.path.isdir(os.path.join(args.source,'llvm','tools','swift')):
       args.lldb = args.source
       args.llvm = os.path.join(args.source,'llvm')
       args.clang = os.path.join(args.source,'llvm','tools','clang')
       args.swift = os.path.join(args.source,'llvm','tools','swift')

def init_libclang(src_path, lib_path):
  def look_for_node(cursor, f):
    if not cursor: return None
    if not f: return None
    if f(cursor): return cursor
    for c in cursor.get_children():
      w = look_for_node(c, f)
      if w: return w
    return None
    
  def printtree(node, depth=0):
    d = '  ' * depth
    print('%s%s (%s)') % (d, node.pretty_print(), node.kind)
    for c in node.get_children():
      printtree(c, depth+1)

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
  clang.cindex.Cursor.pretty_print = lambda self: (self.spelling or self.displayname)
  clang.cindex.Cursor.search = look_for_node
  clang.cindex.Cursor.dump = lambda self: printtree(self, depth=0)
  return True

def main():
  args = get_args()
  if not args.sdk:
    args.sdk = subprocess.check_output('xcrun --sdk macosx --show-sdk-path', shell=True).strip()
  detect_source_layout(args)
  src_path = os.path.join(args.clang,'bindings','python')
  lib_path = os.path.join(args.build,'llvm-macosx-x86_64','lib')
  if not init_libclang(src_path, lib_path):
    print('libclang initialization failed - please try again')

  index = Index()

  macros = [Macro(x) for x in ['__STDC_CONSTANT_MACROS','__STDC_LIMIT_MACROS']]
  includes = [IncludePath(x) for x in [
    os.path.join(args.llvm,'include'),
    os.path.join(args.clang,'include'),
    os.path.join(args.swift,'include'),
    os.path.join(args.lldb,'include'),
    os.path.join(args.lldb,'source'),
    os.path.join(args.build,'llvm-macosx-x86_64','include'),
    os.path.join(args.build,'llvm-macosx-x86_64','tools','clang','include'),
    os.path.join(args.build,'swift-macosx-x86_64','include'),
  ]]
  lang = [CPP11()]
  sdk = [SDKRoot(args.sdk)]
  parser = index.parse(
    os.path.join(args.lldb,'source','Symbol','SwiftASTContext.cpp'),
    macros+lang+sdk+includes)
  
  def search_lambda(cursor):
    if not (cursor.kind == clang.cindex.CursorKind.CXX_METHOD): return False
    parent = cursor.semantic_parent
    if not ((parent.spelling or parent.displayname) == 'SwiftASTContext'): return False
    if not (cursor.is_definition()): return False
    if (cursor.is_static_method()): return False
    return True

  methods = parser.find_if(search_lambda)

  FAIL = 0

  def emit_fail(method):
    print('%s:%s:%s: error: %s not found on method \'%s\'; consider adding' %
      (os.path.basename(method.location.file.name),
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
    if not compound_stmt: return False
    do_stmt = compound_stmt.search(look_for_DO_STMT)
    if not do_stmt: return False
    call_expr = do_stmt.search(look_for_CALL_EXPR)
    if not call_expr: return False
    member_ref_expr = call_expr.search(look_for_MEMBER_REF_EXPR)
    if not member_ref_expr: return False
    if member_ref_expr.pretty_print() == 'HasFatalErrors': return True
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
    unsafe = method.search(lambda c: look_for_IVAR(c) or look_for_METHOD(c))
    return (unsafe is None)

  whitelist = [
    'GetPluginName',
    'GetPluginVersion',
    'HasFatalErrors',
    'GetFatalErrors'
  ]

  for method in methods:
    if method.pretty_print() in whitelist: continue
    try:
      if not scan(method) and not is_safe(method):
        emit_fail(method)
        FAIL += 1
    except Exception as e:
      print(e)
      emit_fail(method)
      FAIL += 1
    
  return FAIL

if main() > 0: sys.exit(1)
