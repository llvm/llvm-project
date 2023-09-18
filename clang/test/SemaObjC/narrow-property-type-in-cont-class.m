// RUN: %clang_cc1 -x objective-c -fsyntax-only -verify %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -verify %s

@interface NSArray @end

@interface NSMutableArray : NSArray
@end

@interface GKTurnBasedMatchMakerKVO
@property(nonatomic,readonly,retain) NSArray* outline;
@property(nonatomic,readonly,retain) NSMutableArray* err_outline; // expected-note {{property declared here}}
@end

@interface GKTurnBasedMatchMakerKVO ()
@property(nonatomic,readwrite,retain) NSMutableArray* outline;
@property(nonatomic,readwrite,retain) NSArray* err_outline; // expected-error {{type of property 'NSArray *' in class extension does not match property type in primary class}}
@end

