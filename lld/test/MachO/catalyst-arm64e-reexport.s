# REQUIRES: aarch64

## When linking a macCatalyst arm64 output, lld should tolerate implicitly
## linked re-exported dylibs that only declare arm64e-macos targets, even if targeting arm64.
# arm64 and arm64e are ABI compatible (same CPU type), and ld64 accepts this.

# RUN: rm -rf %t; split-file --no-leading-lines %s %t

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-ios13.15.0-macabi -o %t/test.o /dev/null

# RUN: %no-arg-lld -syslibroot %t/sdk -lSystem -dylib -arch arm64 \
# RUN:     -platform_version mac-catalyst 13.15.0 14.0 %t/test.o -o /dev/null

#--- sdk/usr/lib/libSystem.tbd
--- !tapi-tbd
tbd-version:      4
targets:          [ arm64-macos, arm64-maccatalyst ]
install-name:     '/usr/lib/libSystem.dylib'
current-version:  0001.001.1
reexported-libraries:
  - targets:      [ arm64-macos, arm64-maccatalyst ]
    libraries:    [ '/usr/lib/system/libsystem_eligibility.dylib' ]
--- !tapi-tbd
tbd-version:      4
targets:          [ arm64e-macos ]
install-name:     '/usr/lib/system/libsystem_eligibility.dylib'
current-version:  0001.001.1
parent-umbrella:
  - targets:      [ arm64e-macos ]
    umbrella:     System
exports:
  - targets:      [ arm64e-macos ]
    symbols:      [ _os_eligibility_get_domain_answer ]
...
