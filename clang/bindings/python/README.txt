//===----------------------------------------------------------------------===//
// Clang Python Bindings
//===----------------------------------------------------------------------===//

This directory implements Python bindings for Clang.

You may need to alter LD_LIBRARY_PATH so that the Clang library can be
found. The unit tests are designed to be run with 'nosetests'. For example:
--
$ env PYTHONPATH=$(echo ~/llvm/tools/clang/bindings/python/) \
      LD_LIBRARY_PATH=$(llvm-config --libdir) \
  nosetests -v
tests.cindex.test_index.test_create ... ok
...

OK
--
