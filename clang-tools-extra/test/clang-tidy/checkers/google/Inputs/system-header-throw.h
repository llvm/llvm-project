#pragma clang system_header

#define SYS_THROW(e) @throw e

#define SYS_RAISE [NSException raise:@"example" format:@"fmt"]

