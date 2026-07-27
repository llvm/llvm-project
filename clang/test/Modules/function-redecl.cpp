// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -x c++ -fmodules -emit-module -fmodule-name=TR %t/module.modulemap -o %t/TR.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -x c++ -fmodules -emit-module -fmodule-name=SDECL %t/module.modulemap -fmodule-file=%t/TR.pcm -I%t -o %t/SDECL.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -x c++ -fmodules -emit-module -fmodule-name=SDEF %t/module.modulemap -fmodule-file=%t/SDECL.pcm -I%t -o %t/SDEF.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -x c++ -fmodules -emit-module -fmodule-name=SP %t/module.modulemap -fmodule-file=%t/SDEF.pcm -I%t -o %t/SP.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -x c++ -fmodules -emit-module -fmodule-name=MOD %t/module.modulemap -fmodule-file=%t/SP.pcm -I%t -o %t/MOD.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -ast-dump -ast-dump-filter=swap %t/MOD.pcm | FileCheck --strict-whitespace --match-full-lines %s

//--- module.modulemap
module TR {
  header "traits.h"
}
module SDECL {
  header "swapdecl.h"
}
module SDEF {
  header "swapdef.h"
  export *
}
module SP {
  header "sp.h"
}
module MOD {
  header "mod.h"
}

//--- traits.h
namespace ns {}

//--- swapdecl.h
#include "traits.h"
namespace ns {
  template <class> void swap();
}

//--- swapdef.h
#include "swapdecl.h"
namespace ns {
  template <class> void swap() {}
} // namespace ns

//--- sp.h
#include "swapdef.h"
template<class> void sp() { ns::swap<int>(); }

//--- mod.h
#include "sp.h"
void use() {
  sp<int>();
}

// CHECK-LABEL:Dumping ns::swap:
// CHECK-NEXT:FunctionTemplateDecl [[FTD1:0x[a-z0-9]+]] <{{.+}}/swapdecl.h:3:3, col:30> col:25 imported in SDECL hidden swap external-linkage
// CHECK-NEXT:|-TemplateTypeParmDecl 0x{{.+}} <col:13> col:18 imported in SDECL hidden class depth 0 index 0
// CHECK-NEXT:|-FunctionDecl [[FD1:0x[a-z0-9]+]] <col:20, col:30> col:25 imported in SDECL hidden swap 'void ()'
// CHECK-NEXT:|-FunctionDecl [[FD2:0x[a-z0-9]+]] prev [[FD3:0x[a-z0-9]+]] <{{.+}}/swapdef.h:3:20, col:33> col:25 imported in SDEF hidden used swap 'void ()' implicit_instantiation instantiated_from [[FD4:0x[a-z0-9]+]] external-linkage
// CHECK-NEXT:| |-TemplateArgument type 'int'
// CHECK-NEXT:| | `-BuiltinType 0x{{.+}} 'int'
// CHECK-NEXT:| `-CompoundStmt 0x{{.+}} <col:32, col:33>
// CHECK-NEXT:`-FunctionDecl [[FD3]] <{{.+}}/swapdecl.h:3:20, col:30> col:25 imported in SDECL hidden used swap 'void ()' implicit_instantiation instantiated_from [[FD1]] external-linkage
// CHECK-NEXT:  `-TemplateArgument type 'int'
// CHECK-NEXT:    `-BuiltinType 0x{{.+}} 'int'

// CHECK-LABEL:Dumping ns::swap:
// CHECK-NEXT:FunctionTemplateDecl 0x{{.+}} prev [[FTD1]] <{{.+}}/swapdef.h:3:3, col:33> col:25 imported in SDEF hidden swap external-linkage
// CHECK-NEXT:|-TemplateTypeParmDecl 0x{{.+}} <col:13> col:18 imported in SDEF hidden class depth 0 index 0
// CHECK-NEXT:|-FunctionDecl [[FD4]] prev [[FD1]] <col:20, col:33> col:25 imported in SDEF hidden swap 'void ()'
// CHECK-NEXT:| `-CompoundStmt 0x{{.+}} <col:32, col:33>
// CHECK-NEXT:|-Function [[FD2]] 'swap' 'void ()'
// CHECK-NEXT:`-Function [[FD3]] 'swap' 'void ()'
