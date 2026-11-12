// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface NSObject @end

@protocol PLAssetContainer
@property (readonly, nonatomic, retain) id assets;
@end


typedef NSObject <PLAssetContainer> PLAlbum; // expected-note {{previous definition is here}}

@class PLAlbum; // expected-warning {{redefinition of forward class 'PLAlbum' of a typedef name of an object type is ignore}}

@interface PLPhotoBrowserController
{
    PLAlbum *_album;
}
@end

@interface WPhotoViewController:PLPhotoBrowserController
@end

@implementation WPhotoViewController
- (void)_prepareForContracting
{
  (void)_album.assets;
}
@end
