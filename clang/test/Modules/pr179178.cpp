// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang -fmodule-name=//b -Xclang=-fno-cxx-modules -Xclang=-fmodule-map-file-home-is-cwd -xc++ -Xclang=-emit-module -fmodules -fno-implicit-modules -fno-implicit-module-maps -c b.cppmap -o b.pic.pcm
// RUN: %clang -fmodule-name=//c -Xclang=-fno-cxx-modules -Xclang=-fmodule-map-file-home-is-cwd -xc++ -Xclang=-emit-module -fmodules -fno-implicit-modules -fno-implicit-module-maps -Xclang=-fmodule-file=b.pic.pcm -c c.cppmap -o c.pic.pcm
// RUN: %clang -fmodule-name=//d -Xclang=-fno-cxx-modules -Xclang=-fmodule-map-file-home-is-cwd -xc++ -Xclang=-emit-module -fmodules -fno-implicit-modules -fno-implicit-module-maps -Xclang=-fmodule-file=c.pic.pcm -c d.cppmap -o d.pic.pcm
// RUN: %clang -fmodule-name=//e -Xclang=-fno-cxx-modules -Xclang=-fmodule-map-file-home-is-cwd -xc++ -Xclang=-emit-module -fmodules -fno-implicit-modules -fno-implicit-module-maps -Xclang=-fmodule-file=d.pic.pcm -c e.cppmap -o e.pic.pcm
// RUN: %clang -fmodule-name=//f -Xclang=-fno-cxx-modules -Xclang=-fmodule-map-file-home-is-cwd -xc++ -Xclang=-emit-module -fmodules -fno-implicit-modules -fno-implicit-module-maps -Xclang=-fmodule-file=e.pic.pcm -c f.cppmap -o f.pic.pcm
// RUN: %clang -fmodule-name=//g -Xclang=-fno-cxx-modules -Xclang=-fmodule-map-file-home-is-cwd -xc++ -Xclang=-emit-module -fmodules -fno-implicit-modules -fno-implicit-module-maps -Xclang=-fmodule-file=f.pic.pcm -c g.cppmap -o g.pic.pcm
// RUN: %clang -fmodule-name=//h -Xclang=-fno-cxx-modules -Xclang=-fmodule-map-file-home-is-cwd -xc++ -Xclang=-emit-module -fmodules -fno-implicit-modules -fno-implicit-module-maps -Xclang=-fmodule-file=g.pic.pcm -c h.cppmap -o h.pic.pcm
// RUN: %clang -fmodule-name=//i -Xclang=-fno-cxx-modules -Xclang=-fmodule-map-file-home-is-cwd -xc++ -Xclang=-emit-module -fmodules -fno-implicit-modules -fno-implicit-module-maps -Xclang=-fmodule-file=h.pic.pcm -c i.cppmap -o i.pic.pcm
// RUN: %clang -fmodule-name=//j -Xclang=-fno-cxx-modules -Xclang=-fmodule-map-file-home-is-cwd -xc++ -Xclang=-emit-module -fmodules -fno-implicit-modules -fno-implicit-module-maps -Xclang=-fmodule-file=i.pic.pcm -c j.cppmap -o j.pic.pcm
// RUN: %clang -fmodule-name=//k -Xclang=-fno-cxx-modules -Xclang=-fmodule-map-file-home-is-cwd -xc++ -Xclang=-emit-module -fmodules -fno-implicit-modules -fno-implicit-module-maps -c k.cppmap -o k.pic.pcm
// RUN: %clang -fmodule-name=//l -Xclang=-fno-cxx-modules -Xclang=-fmodule-map-file-home-is-cwd -xc++ -Xclang=-emit-module -fmodules -fno-implicit-modules -fno-implicit-module-maps -Xclang=-fmodule-file=j.pic.pcm -c l.cppmap -o l.pic.pcm
// RUN: %clang -fmodule-name=//m -Xclang=-fno-cxx-modules -Xclang=-fmodule-map-file-home-is-cwd -xc++ -Xclang=-emit-module -fmodules -fno-implicit-modules -fno-implicit-module-maps -Xclang=-fmodule-file=k.pic.pcm -c m.cppmap -o m.pic.pcm
// RUN: %clang -fmodule-name=//n -Xclang=-fno-cxx-modules -Xclang=-fmodule-map-file-home-is-cwd -xc++ -Xclang=-emit-module -fmodules -fno-implicit-modules -fno-implicit-module-maps -Xclang=-fmodule-file=l.pic.pcm -Xclang=-fmodule-file=m.pic.pcm -c n.cppmap -o n.pic.pcm
// RUN: %clang -fmodule-name=//o -Xclang=-fno-cxx-modules -Xclang=-fmodule-map-file-home-is-cwd -xc++ -Xclang=-emit-module -fmodules -fno-implicit-modules -fno-implicit-module-maps -Xclang=-fmodule-file=n.pic.pcm -c o.cppmap -o o.pic.pcm
// RUN: %clang -Xclang=-fno-cxx-modules -Xclang=-fmodule-map-file-home-is-cwd -fmodules -fno-implicit-modules -fno-implicit-module-maps -Xclang=-fmodule-file=o.pic.pcm -c a.cc -o a.o

//--- a.cc
#include "k.h"
namespace base {
namespace internal {}  
}  
REGISTER_MODULE_INITIALIZER(, );


//--- b.cppmap
module "//b" {
    header "b.h"
}


//--- b.h
namespace base {}  


//--- c.cppmap
module "//c" {
}


//--- d.cppmap
module "//d" {
}


//--- e.cppmap
module "//e" {
}


//--- f.cppmap
module "//f" {
}


//--- g.cppmap
module "//g" {
}


//--- h.cppmap
module "//h" {
}


//--- i.cppmap
module "//i" {
}


//--- j.cppmap
module "//j" {
}


//--- k.cppmap
module "//k" {
    header "k.h"
}


//--- k.h
namespace base {
namespace internal {
struct LiteralTag ;
}  
}  
class FooInitializer ;
#define REGISTER_INITIALIZER(type, name, body)              void foo_init__() ;   FooInitializer initializer_name( base::internal::LiteralTag\
      foo_init__)
#define REGISTER_MODULE_INITIALIZER(name, body) REGISTER_INITIALIZER(, , )


//--- l.cppmap
module "//l" {
}


//--- m.cppmap
module "//m" {
}


//--- n.cppmap
module "//n" {
    header "n.h"
}


//--- n.h
namespace base {}  


//--- o.cppmap
module "//o" {
}
