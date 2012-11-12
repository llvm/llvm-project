from clang.cindex import *
import os

kInputsDir = os.path.join(os.path.dirname(__file__), 'INPUTS')

def test_create():
    index = Index.create()

# FIXME: test Index.read

def test_parse():
    index = Index.create()
    assert isinstance(index, Index)
    tu = index.parse(os.path.join(kInputsDir, 'hello.cpp'))
    assert isinstance(tu, TranslationUnit)
