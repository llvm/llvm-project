struct X { int x; };
void z(int);
typedef struct t TYPEDEF;

void foo() {
  int y = 17;
  // RUN: %clang_cc1 -fsyntax-only -fcxx-exceptions -code-completion-patterns -code-completion-at=%s:6:14 -std=gnu++98 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
  // CHECK-CC1: COMPLETION: bool
  // CHECK-CC1-NEXT: COMPLETION: char
  // CHECK-CC1-NEXT: COMPLETION: class
  // CHECK-CC1-NEXT: COMPLETION: const
  // CHECK-CC1-NEXT: COMPLETION: Pattern : const_cast<<#type#>>(<#expression#>)
  // CHECK-CC1: COMPLETION: Pattern : [#void#]delete <#expression#>
  // CHECK-CC1-NEXT: COMPLETION: Pattern : [#void#]delete [] <#expression#>
  // CHECK-CC1-NEXT: COMPLETION: Pattern : do{<#statements#>
  // CHECK-CC1: COMPLETION: double
  // CHECK-CC1-NEXT: COMPLETION: Pattern : dynamic_cast<<#type#>>(<#expression#>)
  // CHECK-CC1-NEXT: COMPLETION: enum
  // CHECK-CC1-NEXT: COMPLETION: extern
  // CHECK-CC1-NEXT: COMPLETION: Pattern : [#bool#]false
  // CHECK-CC1-NEXT: COMPLETION: float
  // CHECK-CC1-NEXT: COMPLETION: foo : [#void#]foo()
  // CHECK-CC1-NEXT: COMPLETION: Pattern : for(<#init-statement#>;<#condition#>;<#inc-expression#>){
  // CHECK-CC1: COMPLETION: Pattern : goto <#label#>;
  // CHECK-CC1-NEXT: COMPLETION: Pattern : if(<#condition#>){<#statements#>
  // CHECK-CC1: COMPLETION: int
  // CHECK-CC1-NEXT: COMPLETION: long
  // CHECK-CC1-NEXT: COMPLETION: Pattern : new <#type#>(<#expressions#>)
  // CHECK-CC1-NEXT: COMPLETION: Pattern : new <#type#>[<#size#>](<#expressions#>)
  // CHECK-CC1-NEXT: COMPLETION: operator
  // CHECK-CC1-NEXT: COMPLETION: Pattern : reinterpret_cast<<#type#>>(<#expression#>)
  // CHECK-CC1-NEXT: COMPLETION: Pattern : return;
  // CHECK-CC1-NEXT: COMPLETION: short
  // CHECK-CC1-NEXT: COMPLETION: signed
  // CHECK-CC1-NEXT: COMPLETION: Pattern : [#size_t#]sizeof(<#expression-or-type#>)
  // CHECK-CC1-NEXT: COMPLETION: static
  // CHECK-CC1-NEXT: COMPLETION: Pattern : static_cast<<#type#>>(<#expression#>)
  // CHECK-CC1-NEXT: COMPLETION: struct
  // CHECK-CC1-NEXT: COMPLETION: Pattern : switch(<#condition#>){
  // CHECK-CC1: COMPLETION: t : t
  // CHECK-CC1-NEXT: COMPLETION: Pattern : [#void#]throw <#expression#>
  // CHECK-CC1-NEXT: COMPLETION: Pattern : [#bool#]true
  // CHECK-CC1-NEXT: COMPLETION: Pattern : try{<#statements#>
  // CHECK-CC1: COMPLETION: TYPEDEF : TYPEDEF
  // CHECK-CC1-NEXT: COMPLETION: Pattern : typedef <#type#> <#name#>
  // CHECK-CC1-NEXT: COMPLETION: Pattern : [#std::type_info#]typeid(<#expression-or-type#>)
  // CHECK-CC1-NEXT: COMPLETION: Pattern : typename <#qualifier#>::<#name#>
  // CHECK-CC1-NEXT: COMPLETION: Pattern : typeof <#expression#>
  // CHECK-CC1-NEXT: COMPLETION: Pattern : typeof(<#type#>)
  // CHECK-CC1-NEXT: COMPLETION: union
  // CHECK-CC1-NEXT: COMPLETION: unsigned
  // CHECK-CC1-NEXT: COMPLETION: Pattern : using namespace <#identifier#>;
  // CHECK-CC1-NEXT: COMPLETION: void
  // CHECK-CC1-NEXT: COMPLETION: volatile
  // CHECK-CC1-NEXT: COMPLETION: wchar_t
  // CHECK-CC1-NEXT: COMPLETION: Pattern : while(<#condition#>){<#statements#>
  // CHECK-CC1: COMPLETION: X : X
  // CHECK-CC1-NEXT: COMPLETION: y : [#int#]y
  // CHECK-CC1-NEXT: COMPLETION: z : [#void#]z(<#int#>)

  // RUN: %clang_cc1 -fsyntax-only  -code-completion-patterns -code-completion-at=%s:4:1 -std=gnu++98 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
  // CHECK-CC2: COMPLETION: Pattern : asm(<#string-literal#>)
  // CHECK-CC2-NEXT: COMPLETION: bool
  // CHECK-CC2-NEXT: COMPLETION: char
  // CHECK-CC2-NEXT: COMPLETION: class
  // CHECK-CC2-NEXT: COMPLETION: const
  // CHECK-CC2-NEXT: COMPLETION: double
  // CHECK-CC2-NEXT: COMPLETION: enum
  // CHECK-CC2-NEXT: COMPLETION: extern
  // CHECK-CC2-NEXT: COMPLETION: float
  // CHECK-CC2-NEXT: COMPLETION: inline
  // CHECK-CC2-NEXT: COMPLETION: int
  // CHECK-CC2-NEXT: COMPLETION: long
  // CHECK-CC2-NEXT: COMPLETION: Pattern : namespace <#identifier#>{<#declarations#>
  // CHECK-CC2: COMPLETION: Pattern : namespace <#name#> = <#namespace#>;
  // CHECK-CC2-NEXT: COMPLETION: operator
  // CHECK-CC2-NEXT: COMPLETION: short
  // CHECK-CC2-NEXT: COMPLETION: signed
  // CHECK-CC2-NEXT: COMPLETION: static
  // CHECK-CC2-NEXT: COMPLETION: struct
  // CHECK-CC2-NEXT: COMPLETION: t : t
  // CHECK-CC2-NEXT: COMPLETION: Pattern : template <#declaration#>
  // CHECK-CC2-NEXT: COMPLETION: Pattern : template<<#parameters#>>
  // CHECK-CC2-NEXT: COMPLETION: TYPEDEF : TYPEDEF
  // CHECK-CC2-NEXT: COMPLETION: Pattern : typedef <#type#> <#name#>
  // CHECK-CC2-NEXT: COMPLETION: Pattern : typename <#qualifier#>::<#name#>
  // CHECK-CC2-NEXT: COMPLETION: Pattern : typeof <#expression#>
  // CHECK-CC2-NEXT: COMPLETION: Pattern : typeof(<#type#>)
  // CHECK-CC2-NEXT: COMPLETION: union
  // CHECK-CC2-NEXT: COMPLETION: unsigned
  // CHECK-CC2-NEXT: COMPLETION: Pattern : using namespace <#identifier#>;
  // CHECK-CC2-NEXT: COMPLETION: Pattern : using <#qualifier#>::<#name#>;
  // CHECK-CC2-NEXT: COMPLETION: void
  // CHECK-CC2-NEXT: COMPLETION: volatile
  // CHECK-CC2-NEXT: COMPLETION: wchar_t
  // CHECK-CC2-NEXT: COMPLETION: X : X

  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:1:19 -std=gnu++98 %s -o - | FileCheck -check-prefix=CHECK-CC3 %s
  // CHECK-CC3: COMPLETION: bool
  // CHECK-CC3-NEXT: COMPLETION: char
  // CHECK-CC3-NEXT: COMPLETION: class
  // CHECK-CC3-NEXT: COMPLETION: const
  // CHECK-CC3-NEXT: COMPLETION: double
  // CHECK-CC3-NEXT: COMPLETION: enum
  // CHECK-CC3-NEXT: COMPLETION: explicit
  // CHECK-CC3-NEXT: COMPLETION: extern
  // CHECK-CC3-NEXT: COMPLETION: float
  // CHECK-CC3-NEXT: COMPLETION: friend
  // CHECK-CC3-NEXT: COMPLETION: inline
  // CHECK-CC3-NEXT: COMPLETION: int
  // CHECK-CC3-NEXT: COMPLETION: long
  // CHECK-CC3-NEXT: COMPLETION: mutable
  // CHECK-CC3-NEXT: COMPLETION: operator
  // CHECK-CC3-NEXT: COMPLETION: Pattern : private: 
  // CHECK-CC3-NEXT: COMPLETION: Pattern : protected: 
  // CHECK-CC3-NEXT: COMPLETION: Pattern : public: 
  // CHECK-CC3-NEXT: COMPLETION: short
  // CHECK-CC3-NEXT: COMPLETION: signed
  // CHECK-CC3-NEXT: COMPLETION: static
  // CHECK-CC3-NEXT: COMPLETION: struct
  // CHECK-CC3-NEXT: COMPLETION: Pattern : template<<#parameters#>>
  // CHECK-CC3-NEXT: COMPLETION: Pattern : typedef <#type#> <#name#>
  // CHECK-CC3-NEXT: COMPLETION: Pattern : typename <#qualifier#>::<#name#>
  // CHECK-CC3-NEXT: COMPLETION: Pattern : typeof <#expression#>
  // CHECK-CC3-NEXT: COMPLETION: Pattern : typeof(<#type#>)
  // CHECK-CC3-NEXT: COMPLETION: union
  // CHECK-CC3-NEXT: COMPLETION: unsigned
  // CHECK-CC3-NEXT: COMPLETION: Pattern : using <#qualifier#>::<#name#>;
  // CHECK-CC3-NEXT: COMPLETION: virtual
  // CHECK-CC3-NEXT: COMPLETION: void
  // CHECK-CC3-NEXT: COMPLETION: volatile
  // CHECK-CC3-NEXT: COMPLETION: wchar_t
  // CHECK-CC3-NEXT: COMPLETION: X : X

  // RUN: %clang_cc1 -fsyntax-only -fcxx-exceptions -code-completion-patterns -code-completion-at=%s:6:11 -std=gnu++98 %s -o - | FileCheck -check-prefix=CHECK-CC4 %s
  // CHECK-CC4: COMPLETION: bool
  // CHECK-CC4-NEXT: COMPLETION: char
  // CHECK-CC4-NEXT: COMPLETION: class
  // CHECK-CC4-NEXT: COMPLETION: const
  // CHECK-CC4-NEXT: COMPLETION: Pattern : const_cast<<#type#>>(<#expression#>)
  // CHECK-CC4-NEXT: COMPLETION: Pattern : [#void#]delete <#expression#>
  // CHECK-CC4-NEXT: COMPLETION: Pattern : [#void#]delete [] <#expression#>
  // CHECK-CC4-NEXT: COMPLETION: double
  // CHECK-CC4-NEXT: COMPLETION: Pattern : dynamic_cast<<#type#>>(<#expression#>)
  // CHECK-CC4-NEXT: COMPLETION: enum
  // CHECK-CC4-NEXT: COMPLETION: Pattern : [#bool#]false
  // CHECK-CC4-NEXT: COMPLETION: float
  // CHECK-CC4-NEXT: COMPLETION: foo : [#void#]foo()
  // CHECK-CC4-NEXT: COMPLETION: int
  // CHECK-CC4-NEXT: COMPLETION: long
  // CHECK-CC4-NEXT: COMPLETION: Pattern : new <#type#>(<#expressions#>)
  // CHECK-CC4-NEXT: COMPLETION: Pattern : new <#type#>[<#size#>](<#expressions#>)
  // CHECK-CC4-NEXT: COMPLETION: operator
  // CHECK-CC4-NEXT: COMPLETION: Pattern : reinterpret_cast<<#type#>>(<#expression#>)
  // CHECK-CC4-NEXT: COMPLETION: short
  // CHECK-CC4-NEXT: COMPLETION: signed
  // CHECK-CC4-NEXT: COMPLETION: Pattern : [#size_t#]sizeof(<#expression-or-type#>)
  // CHECK-CC4-NEXT: COMPLETION: Pattern : static_cast<<#type#>>(<#expression#>)
  // CHECK-CC4-NEXT: COMPLETION: struct
  // CHECK-CC4-NEXT: COMPLETION: t : t
  // CHECK-CC4-NEXT: COMPLETION: Pattern : [#void#]throw <#expression#>
  // CHECK-CC4-NEXT: COMPLETION: Pattern : [#bool#]true
  // CHECK-CC4-NEXT: COMPLETION: TYPEDEF : TYPEDEF
  // CHECK-CC4-NEXT: COMPLETION: Pattern : [#std::type_info#]typeid(<#expression-or-type#>)
  // CHECK-CC4-NEXT: COMPLETION: Pattern : typename <#qualifier#>::<#name#>
  // CHECK-CC4-NEXT: COMPLETION: Pattern : typeof <#expression#>
  // CHECK-CC4-NEXT: COMPLETION: Pattern : typeof(<#type#>)
  // CHECK-CC4-NEXT: COMPLETION: union
  // CHECK-CC4-NEXT: COMPLETION: unsigned
  // CHECK-CC4-NEXT: COMPLETION: void
  // CHECK-CC4-NEXT: COMPLETION: volatile
  // CHECK-CC4-NEXT: COMPLETION: wchar_t
  // CHECK-CC4-NEXT: COMPLETION: X : X
  // CHECK-CC4-NEXT: COMPLETION: z : [#void#]z(<#int#>)

  // RUN: %clang_cc1 -fsyntax-only -fno-rtti -code-completion-patterns -code-completion-at=%s:6:14 -std=gnu++98 %s -o - | FileCheck -check-prefix=CHECK-NO-RTTI %s
  // CHECK-NO-RTTI: COMPLETION: bool
  // CHECK-NO-RTTI-NEXT: COMPLETION: char
  // CHECK-NO-RTTI-NEXT: COMPLETION: class
  // CHECK-NO-RTTI-NEXT: COMPLETION: const
  // CHECK-NO-RTTI-NEXT: COMPLETION: Pattern : const_cast<<#type#>>(<#expression#>)
  // CHECK-NO-RTTI: COMPLETION: Pattern : [#void#]delete <#expression#>
  // CHECK-NO-RTTI-NEXT: COMPLETION: Pattern : [#void#]delete [] <#expression#>
  // CHECK-NO-RTTI-NEXT: COMPLETION: Pattern : do{<#statements#>
  // CHECK-NO-RTTI: COMPLETION: double
  // CHECK-NO-RTTI-NOT: dynamic_cast
  // CHECK-NO-RTTI: COMPLETION: enum
  // CHECK-NO-RTTI-NEXT: COMPLETION: extern
  // CHECK-NO-RTTI-NEXT: COMPLETION: Pattern : [#bool#]false
  // CHECK-NO-RTTI-NEXT: COMPLETION: float
  // CHECK-NO-RTTI-NEXT: COMPLETION: foo : [#void#]foo()
  // CHECK-NO-RTTI-NEXT: COMPLETION: Pattern : for(<#init-statement#>;<#condition#>;<#inc-expression#>){
  // CHECK-NO-RTTI: COMPLETION: Pattern : goto <#label#>;
  // CHECK-NO-RTTI-NEXT: COMPLETION: Pattern : if(<#condition#>){<#statements#>
  // CHECK-NO-RTTI: COMPLETION: int
  // CHECK-NO-RTTI-NEXT: COMPLETION: long
  // CHECK-NO-RTTI-NEXT: COMPLETION: Pattern : new <#type#>(<#expressions#>)
  // CHECK-NO-RTTI-NEXT: COMPLETION: Pattern : new <#type#>[<#size#>](<#expressions#>)
  // CHECK-NO-RTTI-NEXT: COMPLETION: operator
  // CHECK-NO-RTTI-NEXT: COMPLETION: Pattern : reinterpret_cast<<#type#>>(<#expression#>)
  // CHECK-NO-RTTI-NEXT: COMPLETION: Pattern : return;
  // CHECK-NO-RTTI-NEXT: COMPLETION: short
  // CHECK-NO-RTTI-NEXT: COMPLETION: signed
  // CHECK-NO-RTTI-NEXT: COMPLETION: Pattern : [#size_t#]sizeof(<#expression-or-type#>)
  // CHECK-NO-RTTI-NEXT: COMPLETION: static
  // CHECK-NO-RTTI-NEXT: COMPLETION: Pattern : static_cast<<#type#>>(<#expression#>)
  // CHECK-NO-RTTI-NEXT: COMPLETION: struct
  // CHECK-NO-RTTI-NEXT: COMPLETION: Pattern : switch(<#condition#>){
  // CHECK-NO-RTTI: COMPLETION: t : t
  // CHECK-NO-RTTI-NOT: throw
  // CHECK-NO-RTTI: COMPLETION: Pattern : [#bool#]true
  // CHECK-NO-RTTI-NOT: try
  // CHECK-NO-RTTI: COMPLETION: TYPEDEF : TYPEDEF
  // CHECK-NO-RTTI-NEXT: COMPLETION: Pattern : typedef <#type#> <#name#>
  // CHECK-NO-RTTI-NOT: typeid
  // CHECK-NO-RTTI: COMPLETION: Pattern : typename <#qualifier#>::<#name#>
  // CHECK-NO-RTTI-NEXT: COMPLETION: Pattern : typeof <#expression#>
  // CHECK-NO-RTTI-NEXT: COMPLETION: Pattern : typeof(<#type#>)
  // CHECK-NO-RTTI-NEXT: COMPLETION: union
  // CHECK-NO-RTTI-NEXT: COMPLETION: unsigned
  // CHECK-NO-RTTI-NEXT: COMPLETION: Pattern : using namespace <#identifier#>;
  // CHECK-NO-RTTI-NEXT: COMPLETION: void
  // CHECK-NO-RTTI-NEXT: COMPLETION: volatile
  // CHECK-NO-RTTI-NEXT: COMPLETION: wchar_t
  // CHECK-NO-RTTI-NEXT: COMPLETION: Pattern : while(<#condition#>){<#statements#>
  // CHECK-NO-RTTI: COMPLETION: X : X
  // CHECK-NO-RTTI-NEXT: COMPLETION: y : [#int#]y
  // CHECK-NO-RTTI-NEXT: COMPLETION: z : [#void#]z(<#int#>)
