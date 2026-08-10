// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

class C {};
bool operator == (C c1, C c2);

bool operator == (C c1, int i);
bool operator == (int i, C c2);

C operator += (C c1, C c2);

enum TextureType { TextureType3D  };

@interface Texture
@property  int textureType;
@property  C c;
@end

template <typename> class Framebuffer {
public:
  Texture **color_attachment;  
  Framebuffer();
};

template <typename T> Framebuffer<T>::Framebuffer() {
  (void)(color_attachment[0].textureType == TextureType3D);
  color_attachment[0].textureType += 1;
  (void)(color_attachment[0].c == color_attachment[0].c);
  (void)(color_attachment[0].c == 1);
  (void)(1 == color_attachment[0].c);
}

void foo() {
  Framebuffer<int>();
}
