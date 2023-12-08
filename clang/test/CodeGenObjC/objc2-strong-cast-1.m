// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fobjc-runtime=macosx-fragile-10.5 -fobjc-gc -emit-llvm -o %t %s
// RUN: %clang_cc1 -x objective-c++ -triple x86_64-unknown-unknown -fobjc-runtime=macosx-fragile-10.5 -fobjc-gc -emit-llvm -o %t %s

@interface I {
  __attribute__((objc_gc(strong))) int *i_IdocumentIDs;
  __attribute__((objc_gc(strong))) long *l_IdocumentIDs;
  __attribute__((objc_gc(strong))) long long *ll_IdocumentIDs;
  __attribute__((objc_gc(strong))) float *IdocumentIDs;
  __attribute__((objc_gc(strong))) double *d_IdocumentIDs;
}
- (void) _getResultsOfMatches;
@end

@implementation I
-(void) _getResultsOfMatches {
    IdocumentIDs[2] = IdocumentIDs[3];
    d_IdocumentIDs[2] = d_IdocumentIDs[3];
    l_IdocumentIDs[2] = l_IdocumentIDs[3];
    ll_IdocumentIDs[2] = ll_IdocumentIDs[3];
    i_IdocumentIDs[2] = i_IdocumentIDs[3];
}

@end

