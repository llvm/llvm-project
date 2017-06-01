// RUN: cd %S

// RUN: %clang_cc1 -emit-pch -x objective-c-header %s -o %t_c00.pch -fno-pch-timestamp
// RUN: %clang_cc1 -emit-pch -x objective-c-header %s -o %t_c00_1.pch -fno-pch-timestamp
// RUN: diff %t_c00.pch %t_c00_1.pch

// RUN: %clang_cc1 -emit-pch -x objective-c-header %s -o %t_c00_2.pch -fno-pch-timestamp
// RUN: diff %t_c00.pch %t_c00_2.pch

// RUN: %clang_cc1 -emit-pch -x objective-c-header %s -o %t_c00_3.pch -fno-pch-timestamp
// RUN: diff %t_c00.pch %t_c00_3.pch

// RUN: %clang_cc1 -emit-pch -x objective-c-header %s -o %t_c00_4.pch -fno-pch-timestamp
// RUN: diff %t_c00.pch %t_c00_4.pch

// RUN: %clang_cc1 -emit-pch -x objective-c-header %s -o %t_c00_5.pch -fno-pch-timestamp
// RUN: diff %t_c00.pch %t_c00_5.pch

@protocol NSObject
- (void)doesNotRecognizeSelector:(SEL)aSelector;
- (id)forwardingTargetForSelector:(SEL)aSelector;
@end
