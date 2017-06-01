// RUN: %clang_cc1 -fsyntax-only -Wno-pragmas -verify %s

#pragma clang attribute push (__attribute__((annotate("test"))), apply_to = any(function, variable))

#pragma clang attribute pop

// Check for contradictions in rules for attribute without a strict subject set:

#pragma clang attribute push (__attribute__((annotate("subRuleContradictions"))), apply_to = any(variable, variable(is_parameter), function(is_member), variable(is_global)))
// expected-error@-1 {{redundant attribute subject matcher sub-rule 'variable(is_parameter)'; 'variable' already matches those declarations}}
// expected-error@-2 {{redundant attribute subject matcher sub-rule 'variable(is_global)'; 'variable' already matches those declarations}}

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((annotate("subRuleContradictions2"))), apply_to = any(function(is_member), function))
// expected-error@-1 {{redundant attribute subject matcher sub-rule 'function(is_member)'; 'function' already matches those declarations}}

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((annotate("subRuleContradictions3"))), apply_to = any(variable, variable(unless(is_parameter))))
// expected-error@-1 {{redundant attribute subject matcher sub-rule 'variable(unless(is_parameter))'; 'variable' already matches those declarations}}

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((annotate("negatedSubRuleContradictions1"))), apply_to = any(variable(is_parameter), variable(unless(is_parameter))))
// expected-error@-1 {{negated attribute subject matcher sub-rule 'variable(unless(is_parameter))' contradicts sub-rule 'variable(is_parameter)'}}

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((annotate("negatedSubRuleContradictions2"))), apply_to = any(variable(unless(is_parameter)), variable(is_thread_local), function, variable(is_global)))
// expected-error@-1 {{negated attribute subject matcher sub-rule 'variable(unless(is_parameter))' contradicts sub-rule 'variable(is_global)'}}
// We have just one error, don't error on 'variable(is_global)'

#pragma clang attribute pop

// Verify the strict subject set verification.

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = any(function))
// No error
#pragma clang attribute pop

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = any(record(unless(is_union)), function, variable))
// No error
#pragma clang attribute pop

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = any(function, variable, record(unless(is_union))))
// No error
#pragma clang attribute pop

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = any(variable, record(unless(is_union)), function))
// No error
#pragma clang attribute pop

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = any(function, record(unless(is_union)), variable, enum))
// expected-error@-1 {{attribute 'abi_tag' can't be applied to 'enum'}}
#pragma clang attribute pop

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = any(enum_constant, function, record(unless(is_union)), variable, variable(is_parameter)))
// expected-error@-1 {{attribute 'abi_tag' can't be applied to 'variable(is_parameter)', and 'enum_constant'}}
#pragma clang attribute pop

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = any(function, record(unless(is_union)), enum))
// expected-error@-1 {{attribute 'abi_tag' can't be applied to 'enum'}}
#pragma clang attribute pop

// Verify the non-strict subject set verification.

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = any(function))

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = variable)

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = any(record(unless(is_union))))

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = any(function, variable))

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = any(variable, record(unless(is_union))))

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = any(record(unless(is_union)), function))

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = any(record(unless(is_union)), function, variable))

#pragma clang attribute pop


#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = any(record(unless(is_union)), function, variable, enum, enum_constant))
// expected-error@-1 {{attribute 'abi_tag' can't be applied to 'enum_constant', and 'enum'}}

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = enum)
// expected-error@-1 {{attribute 'abi_tag' can't be applied to 'enum'}}

#pragma clang attribute pop

// Handle attributes whose subjects are supported only in other language modes:

#pragma clang attribute push(__attribute__((abi_tag("b"))), apply_to = any(namespace, record(unless(is_union)), variable, function))
// 'namespace' is accepted!
#pragma clang attribute pop

#pragma clang attribute push(__attribute__((abi_tag("b"))), apply_to = any(namespace))
// 'namespace' is accepted!
#pragma clang attribute pop

#pragma clang attribute push(__attribute__((objc_subclassing_restricted)), apply_to = objc_interface)
// No error!
#pragma clang attribute pop

#pragma clang attribute push(__attribute__((objc_subclassing_restricted)), apply_to = objc_interface)
// No error!
#pragma clang attribute pop

#pragma clang attribute push(__attribute__((objc_subclassing_restricted)), apply_to = any(objc_interface, objc_protocol))
// expected-error@-1 {{attribute 'objc_subclassing_restricted' can't be applied to 'objc_protocol'}}
#pragma clang attribute pop

#pragma clang attribute push(__attribute__((objc_subclassing_restricted)), apply_to = any(objc_protocol))
// expected-error@-1 {{attribute 'objc_subclassing_restricted' can't be applied to 'objc_protocol'}}
// Don't report an error about missing 'objc_interface' as we aren't parsing
// Objective-C.
#pragma clang attribute pop

#pragma clang attribute push(__attribute__((objc_subclassing_restricted)), apply_to = any(objc_interface, objc_protocol))
// expected-error@-1 {{attribute 'objc_subclassing_restricted' can't be applied to 'objc_protocol'}}
#pragma clang attribute pop

#pragma clang attribute push(__attribute__((objc_subclassing_restricted)), apply_to = any(objc_protocol))
// expected-error@-1 {{attribute 'objc_subclassing_restricted' can't be applied to 'objc_protocol'}}
// Don't report an error about missing 'objc_interface' as we aren't parsing
// Objective-C.
#pragma clang attribute pop

// Use of matchers from other language modes should not cause for attributes
// without subject list:
#pragma clang attribute push (__attribute__((annotate("test"))), apply_to = objc_method)

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((annotate("test"))), apply_to = any(objc_interface, objc_protocol))

#pragma clang attribute pop
