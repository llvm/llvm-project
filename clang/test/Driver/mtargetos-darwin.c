// RUN: %clang -mtargetos=macos11 -arch arm64 -arch x86_64 -c %s -o %t.o -### 2>&1 | FileCheck --check-prefix=MACOS %s
// RUN: %clang -mtargetos=ios14 -arch arm64 -c %s -o %t.o -### 2>&1 | FileCheck --check-prefix=IOS %s
// RUN: %clang -mtargetos=ios14-simulator -arch arm64 -c %s -o %t.o -### 2>&1 | FileCheck --check-prefix=IOS_SIM %s
// RUN: %clang -mtargetos=ios14-macabi -arch arm64 -c %s -o %t.o -### 2>&1 | FileCheck --check-prefix=MACCATALYST %s
// RUN: %clang -mtargetos=tvos14 -arch arm64 -c %s -o %t.o -### 2>&1 | FileCheck --check-prefix=TVOS %s
// RUN: %clang -mtargetos=watchos7.1 -arch arm64 -c %s -o %t.o -### 2>&1 | FileCheck --check-prefix=WATCHOS %s

// RUN: %clang -target arm64-apple-ios14 -mtargetos=ios14 -arch arm64 -c %s -o %t.o -### 2>&1 | FileCheck --check-prefix=NOMIX1 %s
// RUN: %clang -mtargetos=ios14 -arch arm64 -miphoneos-version-min=14 -c %s -o %t.o -### 2>&1 | FileCheck --check-prefix=NOMIX2 %s
// RUN: %clang -mtargetos=darwin20 -arch arm64 -c %s -o %t.o -### 2>&1 | FileCheck --check-prefix=INVALIDOS %s
// RUN: %clang -mtargetos=ios -arch arm64 -c %s -o %t.o -### 2>&1 | FileCheck --check-prefix=NOVERSION %s

// REQUIRES: system-darwin

// MACOS: "-cc1" "-triple" "arm64-apple-macosx11.0.0"
// MACOS-NEXT: "-cc1" "-triple" "x86_64-apple-macosx11.0.0"
// IOS: "-cc1" "-triple" "arm64-apple-ios14.0.0"
// IOS_SIM: "-cc1" "-triple" "arm64-apple-ios14.0.0-simulator"
// MACCATALYST: "-cc1" "-triple" "arm64-apple-ios14.0.0-macabi"
// TVOS: "-cc1" "-triple" "arm64-apple-tvos14.0.0"
// WATCHOS: "-cc1" "-triple" "arm64-apple-watchos7.1.0"

// NOMIX1: error: cannot specify '-mtargetos=ios14' along with '-target arm64-apple-ios14'
// NOMIX2: error: cannot specify '-miphoneos-version-min=14' along with '-mtargetos=ios14'
// INVALIDOS: error: invalid OS value 'darwin20' in '-mtargetos=darwin20'
// NOVERSION: error: invalid version number in '-mtargetos=ios'
