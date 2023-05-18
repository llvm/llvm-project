// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits -Wno-pragma-clang-attribute %s 2>&1 | FileCheck %s

#pragma clang attribute push (annotate)
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:31-[[@LINE-1]]:31}:"__attribute__(("
// CHECK: fix-it:{{.*}}:{[[@LINE-2]]:39-[[@LINE-2]]:39}:"))"
#pragma clang attribute push (annotate(("test")))
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:31-[[@LINE-1]]:31}:"__attribute__(("
// CHECK: fix-it:{{.*}}:{[[@LINE-2]]:49-[[@LINE-2]]:49}:"))"

#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( enum, function, function, namespace, function ))
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:97-[[@LINE-1]]:107}:""
// CHECK: fix-it:{{.*}}:{[[@LINE-2]]:118-[[@LINE-2]]:127}:""

#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( variable(is_global), function, variable(is_global), variable(is_global) ))
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:112-[[@LINE-1]]:133}:""
// CHECK: fix-it:{{.*}}:{[[@LINE-2]]:133-[[@LINE-2]]:153}:""

#pragma clang attribute push (__attribute__((annotate("subRuleContradictions"))), apply_to = any(variable, variable(is_parameter), function(is_member), variable(is_global)))
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:153-[[@LINE-1]]:172}:""
// CHECK: fix-it:{{.*}}:{[[@LINE-2]]:108-[[@LINE-2]]:132}:""

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((annotate("subRuleContradictions2"))), apply_to = any(function(is_member),function))
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:99-[[@LINE-1]]:119}:""

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((annotate("negatedSubRuleContradictions1"))), apply_to = any(variable(is_parameter), variable(unless(is_parameter))))
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:130-[[@LINE-1]]:160}:""
#pragma clang attribute pop

#pragma clang attribute push (__attribute__((annotate("negatedSubRuleContradictions2"))), apply_to = any(variable(unless(is_parameter)), variable(is_thread_local), function, variable(is_global)))
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:106-[[@LINE-1]]:137}:""
#pragma clang attribute pop

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = any(enum, variable))
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:77-[[@LINE-1]]:82}:""
#pragma clang attribute pop

#pragma clang attribute push (__attribute__((abi_tag("a"))))
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:60-[[@LINE-1]]:60}:", apply_to = any(function, namespace, record(unless(is_union)), variable)"
#pragma clang attribute push (__attribute__((abi_tag("a"))) apply_to=function)
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:60-[[@LINE-1]]:60}:", "
#pragma clang attribute push (__attribute__((abi_tag("a"))) = function)
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:60-[[@LINE-1]]:60}:", apply_to"
#pragma clang attribute push (__attribute__((abi_tag("a"))) any(function))
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:60-[[@LINE-1]]:60}:", apply_to = "

#pragma clang attribute push (__attribute__((abi_tag("a"))) 22)
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:60-[[@LINE-1]]:63}:", apply_to = any(function, namespace, record(unless(is_union)), variable)"
#pragma clang attribute push (__attribute__((abi_tag("a"))) function)
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:60-[[@LINE-1]]:69}:", apply_to = any(function, namespace, record(unless(is_union)), variable)"
#pragma clang attribute push (__attribute__((abi_tag("a"))) (function))
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:60-[[@LINE-1]]:71}:", apply_to = any(function, namespace, record(unless(is_union)), variable)"

#pragma clang attribute push (__attribute__((abi_tag("a"))), )
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:61-[[@LINE-1]]:62}:"apply_to = any(function, namespace, record(unless(is_union)), variable)"
#pragma clang attribute push (__attribute__((abi_tag("a"))), = function)
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:61-[[@LINE-1]]:61}:"apply_to"
#pragma clang attribute push (__attribute__((abi_tag("a"))), any(function))
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:61-[[@LINE-1]]:61}:"apply_to = "

#pragma clang attribute push (__attribute__((abi_tag("a"))), 22)
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:61-[[@LINE-1]]:64}:"apply_to = any(function, namespace, record(unless(is_union)), variable)"
#pragma clang attribute push (__attribute__((abi_tag("a"))), 1, 2)
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:61-[[@LINE-1]]:66}:"apply_to = any(function, namespace, record(unless(is_union)), variable)"
#pragma clang attribute push (__attribute__((abi_tag("a"))), function)
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:61-[[@LINE-1]]:70}:"apply_to = any(function, namespace, record(unless(is_union)), variable)"
#pragma clang attribute push (__attribute__((abi_tag("a"))), (function))
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:61-[[@LINE-1]]:72}:"apply_to = any(function, namespace, record(unless(is_union)), variable)"

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to)
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:70-[[@LINE-1]]:70}:" = any(function, namespace, record(unless(is_union)), variable)"
#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to any(function))
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:70-[[@LINE-1]]:70}:" = "

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to 41 (22))
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:70-[[@LINE-1]]:78}:" = any(function, namespace, record(unless(is_union)), variable)"

// Don't give fix-it to attributes without a strict subject set
#pragma clang attribute push (__attribute__((annotate("a"))))
// CHECK-NO: [[@LINE-1]]:61

#pragma clang attribute push (__attribute__((objc_externally_retained)), apply_to)
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:82-[[@LINE-1]]:82}:" = any(function, variable(unless(is_parameter)))"
