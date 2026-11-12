// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -verify -Wno-objc-root-class %s

@class NSMutableDictionary; // expected-note {{receiver is instance of class declared here}}

@interface LaunchdJobs 

@property (nonatomic,retain) NSMutableDictionary *uuids_jobs; // expected-note {{'_uuids_jobs' declared here}}

@end

@implementation LaunchdJobs

-(void)job
{

 [uuids_jobs objectForKey]; // expected-error {{use of undeclared identifier 'uuids_jobs'}} \
                            // expected-warning {{instance method '-objectForKey' not found}}
}


@end

void
doLaunchdJobCPU(void)
{
 [uuids_jobs enumerateKeysAndObjectsUsingBlock]; // expected-error {{use of undeclared identifier 'uuids_jobs'}}
}
