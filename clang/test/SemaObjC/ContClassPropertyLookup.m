// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics

@interface MyObject {
    int _foo;
}
@end

@interface MyObject(whatever)
@property (assign) int foo;
@end

@interface MyObject()
@property (assign) int foo;
@end

@implementation MyObject
@synthesize foo = _foo;
@end

@interface MPMediaItem
@end

@class MPMediaItem;

@interface MPMediaItem ()
@property (nonatomic, readonly) id title;
@end

@interface PodcastEpisodesViewController
@end

@implementation PodcastEpisodesViewController
- (id) Meth {
    MPMediaItem *episode;
    return episode.title;
}
@end
