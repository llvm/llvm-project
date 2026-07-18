// Verify that cc1 -fsycl-is-device rejects C inputs.
// RUN: not %clang_cc1 -fsycl-is-device -triple spirv64-unknown-unknown -x c %s 2>&1 | FileCheck -check-prefix ERR_DEVICE %s
// ERR_DEVICE: error: invalid argument 'C' not allowed with '-fsycl'

// Verify that cc1 -fsycl-is-host rejects C inputs.
// RUN: not %clang_cc1 -fsycl-is-host -triple x86_64-unknown-linux-gnu -x c %s 2>&1 | FileCheck -check-prefix ERR_HOST %s
// ERR_HOST: error: invalid argument 'C' not allowed with '-fsycl'
