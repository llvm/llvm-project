//==--- DiagnosticCommonKinds.td - common diagnostics ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Common Helpers
//===----------------------------------------------------------------------===//

let Component = "Common" in {

// Basic.

def fatal_too_many_errors
  : Error<"too many errors emitted, stopping now">, DefaultFatal; 

def note_declared_at : Note<"declared here">;
def note_previous_definition : Note<"previous definition is here">;
def note_previous_declaration : Note<"previous declaration is here">;
def note_previous_implicit_declaration : Note<
  "previous implicit declaration is here">;
def note_previous_use : Note<"previous use is here">;
def note_duplicate_case_prev : Note<"previous case defined here">;
def note_forward_declaration : Note<"forward declaration of %0">;
def note_type_being_defined : Note<
  "definition of %0 is not complete until the closing '}'">;
/// note_matching - this is used as a continuation of a previous diagnostic,
/// e.g. to specify the '(' when we expected a ')'.
def note_matching : Note<"to match this %0">;

def note_using : Note<"using">;
def note_possibility : Note<"one possibility">;
def note_also_found : Note<"also found">;

// Parse && Lex

let CategoryName = "Lexical or Preprocessor Issue" in {

def err_expected_colon_after_setter_name : Error<
  "method name referenced in property setter attribute "
  "must end with ':'">;
def err_expected_string_literal : Error<"expected string literal "
  "%select{in %1|for diagnostic message in static_assert|"
          "for optional message in 'availability' attribute}0">;
def err_invalid_string_udl : Error<
  "string literal with user-defined suffix cannot be used here">;
def err_invalid_character_udl : Error<
  "character literal with user-defined suffix cannot be used here">;
def err_invalid_numeric_udl : Error<
  "numeric literal with user-defined suffix cannot be used here">;

}

// Parse && Sema

let CategoryName = "Parse Issue" in {

def err_expected : Error<"expected %0">;
def err_expected_either : Error<"expected %0 or %1">;
def err_expected_after : Error<"expected %1 after %0">;

def err_param_redefinition : Error<"redefinition of parameter %0">;
def warn_method_param_redefinition : Warning<"redefinition of method parameter %0">;
def warn_method_param_declaration : Warning<"redeclaration of method parameter %0">,
  InGroup<DuplicateArgDecl>, DefaultIgnore;
def err_invalid_storage_class_in_func_decl : Error<
  "invalid storage class specifier in function declarator">;
def err_expected_namespace_name : Error<"expected namespace name">;
def ext_variadic_templates : ExtWarn<
  "variadic templates are a C++11 extension">, InGroup<CXX11>;
def warn_cxx98_compat_variadic_templates :
  Warning<"variadic templates are incompatible with C++98">,
  InGroup<CXX98Compat>, DefaultIgnore;
def err_default_special_members : Error<
  "only special member functions may be defaulted">;
def err_deleted_non_function : Error<
  "only functions can have deleted definitions">;
def err_module_not_found : Error<"module '%0' not found">, DefaultFatal;
def err_module_not_built : Error<"could not build module '%0'">, DefaultFatal;
def err_module_build_disabled: Error<
  "module '%0' is needed but has not been provided, and implicit use of module "
  "files is disabled">, DefaultFatal;
def err_module_unavailable : Error<
  "module '%0' %select{is incompatible with|requires}1 feature '%2'">;
def err_module_header_missing : Error<
  "%select{|umbrella }0header '%1' not found">;
def err_module_lock_failure : Error<
  "could not acquire lock file for module '%0': %1">, DefaultFatal;
def err_module_lock_timeout : Error<
  "timed out waiting to acquire lock file for module '%0'">, DefaultFatal;
def err_module_cycle : Error<"cyclic dependency in module '%0': %1">, 
  DefaultFatal;
def err_module_prebuilt : Error<
  "error in loading module '%0' from prebuilt module path">, DefaultFatal;
def note_pragma_entered_here : Note<"#pragma entered here">;  
def note_decl_hiding_tag_type : Note<
  "%1 %0 is hidden by a non-type declaration of %0 here">;
def err_attribute_not_type_attr : Error<
  "%0 attribute cannot be applied to types">;
def err_enum_template : Error<"enumeration cannot be a template">;

}

let CategoryName = "Nullability Issue" in {

def warn_nullability_duplicate : Warning<
  "duplicate nullability specifier %0">,
  InGroup<Nullability>;
  
def warn_conflicting_nullability_attr_overriding_ret_types : Warning<
  "conflicting nullability specifier on return types, %0 "
  "conflicts with existing specifier %1">,
  InGroup<Nullability>;

def warn_conflicting_nullability_attr_overriding_param_types : Warning<
  "conflicting nullability specifier on parameter types, %0 "
  "conflicts with existing specifier %1">,
  InGroup<Nullability>;

def err_nullability_conflicting : Error<
  "nullability specifier %0 conflicts with existing specifier %1">;

}

// Sema && Lex
def ext_c99_longlong : Extension<
  "'long long' is an extension when C99 mode is not enabled">,
  InGroup<LongLong>;
def ext_cxx11_longlong : Extension<
  "'long long' is a C++11 extension">,
  InGroup<CXX11LongLong>;
def warn_cxx98_compat_longlong : Warning<
  "'long long' is incompatible with C++98">,
  InGroup<CXX98CompatPedantic>, DefaultIgnore;
def err_integer_literal_too_large : Error<
  "integer literal is too large to be represented in any %select{signed |}0"
  "integer type">;
def ext_integer_literal_too_large_for_signed : ExtWarn<
  "integer literal is too large to be represented in a signed integer type, "
  "interpreting as unsigned">,
  InGroup<ImplicitlyUnsignedLiteral>;
def warn_old_implicitly_unsigned_long : Warning<
  "integer literal is too large to be represented in type 'long', "
  "interpreting as 'unsigned long' per C89; this literal will "
  "%select{have type 'long long'|be ill-formed}0 in C99 onwards">,
  InGroup<C99Compat>;
def warn_old_implicitly_unsigned_long_cxx : Warning<
  "integer literal is too large to be represented in type 'long', "
  "interpreting as 'unsigned long' per C++98; this literal will "
  "%select{have type 'long long'|be ill-formed}0 in C++11 onwards">,
  InGroup<CXX11Compat>;
def ext_old_implicitly_unsigned_long_cxx : ExtWarn<
  "integer literal is too large to be represented in type 'long' and is "
  "subject to undefined behavior under C++98, interpreting as 'unsigned long'; "
  "this literal will %select{have type 'long long'|be ill-formed}0 "
  "in C++11 onwards">,
  InGroup<CXX11Compat>;
def ext_clang_enable_if : Extension<"'enable_if' is a clang extension">,
                          InGroup<GccCompat>;

// SEH
def err_seh_expected_handler : Error<
  "expected '__except' or '__finally' block">;
def err_seh___except_block : Error<
  "%0 only allowed in __except block or filter expression">;
def err_seh___except_filter : Error<
  "%0 only allowed in __except filter expression">;
def err_seh___finally_block : Error<
  "%0 only allowed in __finally block">;

// Sema && AST
def note_invalid_subexpr_in_const_expr : Note<
  "subexpression not valid in a constant expression">;

// Targets

def err_target_unknown_triple : Error<
  "unknown target triple '%0', please use -triple or -arch">;
def err_target_unknown_cpu : Error<"unknown target CPU '%0'">;
def err_target_unknown_abi : Error<"unknown target ABI '%0'">;
def err_target_unsupported_abi : Error<"ABI '%0' is not supported on CPU '%1'">;
def err_target_unsupported_abi_for_triple : Error<
  "ABI '%0' is not supported for '%1'">;
def err_target_unknown_fpmath : Error<"unknown FP unit '%0'">;
def err_target_unsupported_fpmath : Error<
    "the '%0' unit is not supported with this instruction set">;
def err_target_unsupported_unaligned : Error<
  "the %0 sub-architecture does not support unaligned accesses">;
def err_opt_not_valid_with_opt : Error<
  "option '%0' cannot be specified with '%1'">;

// Source manager
def err_cannot_open_file : Error<"cannot open file '%0': %1">, DefaultFatal;
def err_file_modified : Error<
  "file '%0' modified since it was first processed">, DefaultFatal;
def err_unsupported_bom : Error<"%0 byte order mark detected in '%1', but "
  "encoding is not supported">, DefaultFatal;
def err_unable_to_rename_temp : Error<
  "unable to rename temporary '%0' to output file '%1': '%2'">;
def err_unable_to_make_temp : Error<
  "unable to make temporary file: %0">;
  
// Modules
def err_module_format_unhandled : Error<
  "no handler registered for module format '%0'">, DefaultFatal;

// TransformActions
// TODO: Use a custom category name to distinguish rewriter errors.
def err_mt_message : Error<"[rewriter] %0">, SuppressInSystemHeader;
def warn_mt_message : Warning<"[rewriter] %0">;
def note_mt_message : Note<"[rewriter] %0">;

// ARCMigrate
def warn_arcmt_nsalloc_realloc : Warning<"[rewriter] call returns pointer to GC managed memory; it will become unmanaged in ARC">;
def err_arcmt_nsinvocation_ownership : Error<"NSInvocation's %0 is not safe to be used with an object with ownership other than __unsafe_unretained">;

// OpenMP
def err_omp_more_one_clause : Error<
  "directive '#pragma omp %0' cannot contain more than one '%1' clause%select{| with '%3' name modifier| with 'source' dependence}2">;
}
