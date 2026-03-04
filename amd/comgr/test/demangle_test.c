//===- demangle_test.c ----------------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"

int test(const char *MangledName, const char *ExpectedString) {
  amd_comgr_data_t MangledData;
  amd_comgr_data_t DemangledData;
  amd_comgr_status_t Status;

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_BYTES, &MangledData);
  checkError(Status, "amd_comgr_create_data");

  size_t Size = strlen(MangledName);
  Status = amd_comgr_set_data(MangledData, Size, MangledName);
  checkError(Status, "amd_comgr_set_data");

  Status = amd_comgr_demangle_symbol_name(MangledData, &DemangledData);
  checkError(Status, "amd_comgr_demangle_symbol_name");

  size_t DemangledSize = 0;
  Status = amd_comgr_get_data(DemangledData, &DemangledSize, NULL);
  checkError(Status, "amd_comgr_get_data");

  if (DemangledSize != strlen(ExpectedString)) {
    fail("DemangledSize (%zu) does not match ExpectedString size(%zu)\n",
         DemangledSize, ExpectedString);
  }

  char *DemangledName = (char *)calloc(DemangledSize, sizeof(char));
  if (DemangledName == NULL) {
    fail("calloc failed\n");
  }

  Status = amd_comgr_get_data(DemangledData, &DemangledSize, DemangledName);
  checkError(Status, "amd_comgr_get_data");

  if (strncmp(DemangledName, ExpectedString, DemangledSize) != 0) {
    fail(">> expected %s \n >> got %s\n", ExpectedString, DemangledName);
  }

  free(DemangledName);

  Status = amd_comgr_release_data(MangledData);
  checkError(Status, "amd_comgr_release_data");

  Status = amd_comgr_release_data(DemangledData);
  checkError(Status, "amd_comgr_release_data");

  return 0;
}

int main(int argc, char *argv[]) {
  // Tests from llvm/unittests/Demangle/DemangleTest.cpp
  test("_", "_");
  test("_Z3fooi", "foo(int)");
  test("__Z3fooi", "foo(int)");
  test("___Z3fooi_block_invoke", "invocation function for block in foo(int)");
  test("____Z3fooi_block_invoke", "invocation function for block in foo(int)");
  test("?foo@@YAXH@Z", "void __cdecl foo(int)");
  test("foo", "foo");
  test("_RNvC3foo3bar", "foo::bar");
  test("_Z3fooILi79EEbU7_ExtIntIXT_EEi", "bool foo<79>(int _ExtInt<79>)");

  // Some additional test cases.
  test("_Znwm", "operator new(unsigned long)");
  test("_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSERKS4_",
       "std::__cxx11::basic_string<char, std::char_traits<char>, "
       "std::allocator<char>>::operator=(std::__cxx11::basic_string<char, "
       "std::char_traits<char>, std::allocator<char>> const&)");
  test("_ZSt29_Rb_tree_insert_and_rebalancebPSt18_Rb_tree_node_baseS0_RS_",
       "std::_Rb_tree_insert_and_rebalance(bool, std::_Rb_tree_node_base*, "
       "std::_Rb_tree_node_base*, std::_Rb_tree_node_base&)");
  test("_ZSt17__throw_bad_allocv", "std::__throw_bad_alloc()");
  test("_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev",
       "std::__cxx11::basic_string<char, std::char_traits<char>, "
       "std::allocator<char>>::~basic_string()");
  test("_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED1Ev",
       "std::__cxx11::basic_string<char, std::char_traits<char>, "
       "std::allocator<char>>::~basic_string()");
  test("_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base",
       "std::_Rb_tree_increment(std::_Rb_tree_node_base*)");
  test("_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2Ev",
       "std::__cxx11::basic_string<char, std::char_traits<char>, "
       "std::allocator<char>>::basic_string()");
  test("_ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKNSt7__"
       "cxx1112basic_stringIS4_S5_T1_EE",
       "std::basic_ostream<char, std::char_traits<char>>& std::operator<<"
       "<char, std::char_traits<char>, std::allocator<char>"
       ">(std::basic_ostream<char, std::char_traits<char>>&, "
       "std::__cxx11::basic_string<char, std::char_traits<char>, "
       "std::allocator<char>> const&)");
  test("_ZdlPv", "operator delete(void*)");
  test("_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc",
       "std::basic_ostream<char, std::char_traits<char>>& std::operator<<"
       "<std::char_traits<char>>(std::basic_ostream<char, "
       "std::char_traits<char>>&, char const*)");
  test("_ZdlPvm", "operator delete(void*, unsigned long)");
  test("_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base",
       "std::_Rb_tree_decrement(std::_Rb_tree_node_base*)");
  test("_ZNSaIcED1Ev", "std::allocator<char>::~allocator()");
  test("_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1EPKcRKS3_",
       "std::__cxx11::basic_string<char, std::char_traits<char>, "
       "std::allocator<char>>::basic_string(char const*, std::allocator<char> "
       "const&)");
  test("_ZNSt8ios_base4InitC1Ev", "std::ios_base::Init::Init()");
  test("_ZNSolsEi", "std::ostream::operator<<(int)");
  test("_ZNSaIcEC1Ev", "std::allocator<char>::allocator()");
  return 0;
}
