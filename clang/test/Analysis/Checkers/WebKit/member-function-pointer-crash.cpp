// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedLocalVarsChecker -verify %s

#include "mock-types.h"

class RenderStyle;

class FillLayer {
public:
    void ref() const;
    void deref() const;
};

class FillLayersPropertyWrapper {
public:
    typedef const FillLayer& (RenderStyle::*LayersGetter)() const;

private:
    bool canInterpolate(const RenderStyle& from) const
    {
        auto* fromLayer = &(from.*m_layersGetter)();
        // expected-warning@-1{{Local variable 'fromLayer' is uncounted and unsafe}}
        return true;
    }

    LayersGetter m_layersGetter;
};
