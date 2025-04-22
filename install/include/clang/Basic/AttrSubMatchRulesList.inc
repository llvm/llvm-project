/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* List of all attribute subject matching rules that Clang recognizes         *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Attr.td                                                              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef ATTR_MATCH_RULE
#define ATTR_MATCH_RULE(NAME) 
#endif

#ifndef ATTR_MATCH_SUB_RULE
#define ATTR_MATCH_SUB_RULE(Value, Spelling, IsAbstract, Parent, IsNegated) ATTR_MATCH_RULE(Value, Spelling, IsAbstract)
#endif
ATTR_MATCH_RULE(SubjectMatchRule_block, "block", 0)
ATTR_MATCH_RULE(SubjectMatchRule_enum, "enum", 0)
ATTR_MATCH_RULE(SubjectMatchRule_enum_constant, "enum_constant", 0)
ATTR_MATCH_RULE(SubjectMatchRule_field, "field", 0)
ATTR_MATCH_RULE(SubjectMatchRule_function, "function", 0)
ATTR_MATCH_SUB_RULE(SubjectMatchRule_function_is_member, "function(is_member)", 0, attr::SubjectMatchRule_function, 0)
ATTR_MATCH_RULE(SubjectMatchRule_namespace, "namespace", 0)
ATTR_MATCH_RULE(SubjectMatchRule_objc_category, "objc_category", 0)
ATTR_MATCH_RULE(SubjectMatchRule_objc_implementation, "objc_implementation", 0)
ATTR_MATCH_RULE(SubjectMatchRule_objc_interface, "objc_interface", 0)
ATTR_MATCH_RULE(SubjectMatchRule_objc_method, "objc_method", 0)
ATTR_MATCH_SUB_RULE(SubjectMatchRule_objc_method_is_instance, "objc_method(is_instance)", 0, attr::SubjectMatchRule_objc_method, 0)
ATTR_MATCH_RULE(SubjectMatchRule_objc_property, "objc_property", 0)
ATTR_MATCH_RULE(SubjectMatchRule_objc_protocol, "objc_protocol", 0)
ATTR_MATCH_RULE(SubjectMatchRule_record, "record", 0)
ATTR_MATCH_SUB_RULE(SubjectMatchRule_record_not_is_union, "record(unless(is_union))", 0, attr::SubjectMatchRule_record, 1)
ATTR_MATCH_RULE(SubjectMatchRule_hasType_abstract, "hasType", 1)
ATTR_MATCH_SUB_RULE(SubjectMatchRule_hasType_functionType, "hasType(functionType)", 0, attr::SubjectMatchRule_hasType_abstract, 0)
ATTR_MATCH_RULE(SubjectMatchRule_type_alias, "type_alias", 0)
ATTR_MATCH_RULE(SubjectMatchRule_variable, "variable", 0)
ATTR_MATCH_SUB_RULE(SubjectMatchRule_variable_is_thread_local, "variable(is_thread_local)", 0, attr::SubjectMatchRule_variable, 0)
ATTR_MATCH_SUB_RULE(SubjectMatchRule_variable_is_global, "variable(is_global)", 0, attr::SubjectMatchRule_variable, 0)
ATTR_MATCH_SUB_RULE(SubjectMatchRule_variable_is_local, "variable(is_local)", 0, attr::SubjectMatchRule_variable, 0)
ATTR_MATCH_SUB_RULE(SubjectMatchRule_variable_is_parameter, "variable(is_parameter)", 0, attr::SubjectMatchRule_variable, 0)
ATTR_MATCH_SUB_RULE(SubjectMatchRule_variable_not_is_parameter, "variable(unless(is_parameter))", 0, attr::SubjectMatchRule_variable, 1)
#undef ATTR_MATCH_SUB_RULE
#undef ATTR_MATCH_RULE
