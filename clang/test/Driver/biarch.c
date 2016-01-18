// RUN: %clang -target i386--netbsd -m32 %s -### 2> %t
// RUN: grep '"-cc1" "-triple" "i386--netbsd"' %t

// RUN: %clang -target i386--netbsd -m64 %s -### 2> %t
// RUN: grep '"-cc1" "-triple" "x86_64--netbsd"' %t

// RUN: %clang -target x86_64--netbsd -m32 %s -### 2> %t
// RUN: grep '"-cc1" "-triple" "i386--netbsd"' %t

// RUN: %clang -target x86_64--netbsd -m64 %s -### 2> %t
// RUN: grep '"-cc1" "-triple" "x86_64--netbsd"' %t

// RUN: %clang -target armv6--netbsd-eabihf -m32 %s -### 2> %t
// RUN: grep '"-cc1" "-triple" "armv6k--netbsd-eabihf"' %t

// RUN: %clang -target sparcv9--netbsd -m32 %s -### 2> %t
// RUN: grep '"-cc1" "-triple" "sparc--netbsd"' %t

// RUN: %clang -target sparcv9--netbsd -m64 %s -### 2> %t
// RUN: grep '"-cc1" "-triple" "sparcv9--netbsd"' %t

// RUN: %clang -target sparc64--netbsd -m64 %s -### 2> %t
// RUN: grep '"-cc1" "-triple" "sparc64--netbsd"' %t

// RUN: %clang -target sparc--netbsd -m32 %s -### 2> %t
// RUN: grep '"-cc1" "-triple" "sparc--netbsd"' %t

// RUN: %clang -target sparc--netbsd -m64 %s -### 2> %t
// RUN: grep '"-cc1" "-triple" "sparcv9--netbsd"' %t

// RUN: %clang -target sparcel -o foo %s -### 2> %t
// RUN: grep 'gcc\(\.exe\)\?" "-EL" "-o" "foo"' %t

// RUN: %clang -target mips64--netbsd -m32 %s -### 2> %t
// RUN: grep '"-cc1" "-triple" "mips--netbsd"' %t

// RUN: %clang -target mips64--netbsd -m64 %s -### 2> %t
// RUN: grep '"-cc1" "-triple" "mips64--netbsd"' %t

// RUN: %clang -target mips--netbsd -m32 %s -### 2> %t
// RUN: grep '"-cc1" "-triple" "mips--netbsd"' %t

// RUN: %clang -target mips--netbsd -m64 %s -### 2> %t
// RUN: grep '"-cc1" "-triple" "mips64--netbsd"' %t
