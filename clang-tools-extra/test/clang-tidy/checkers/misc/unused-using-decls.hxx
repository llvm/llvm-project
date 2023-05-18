// RUN: %check_clang_tidy %s misc-unused-using-decls %t -- --fix-notes -- -fno-delayed-template-parsing -isystem %S/Inputs

// Verify that we don't generate the warnings on header files.
namespace foo { class Foo {}; }

using foo::Foo;
