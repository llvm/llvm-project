// THIS IS A GENERATED TEST. DO NOT EDIT.
// To regenerate, see clang-doc/gen_test.py docstring.
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"

/// \brief Brief description.
///
/// Extended description that
/// continues onto the next line.
/// 
/// <ul class="test">
///   <li> Testing.
/// </ul>
///
/// \verbatim
/// The description continues.
/// \endverbatim
/// --
/// \param [out] I is a parameter.
/// \param J is a parameter.
/// \return void
void F(int I, int J);

/// Bonus comment on definition
void F(int I, int J) {}
