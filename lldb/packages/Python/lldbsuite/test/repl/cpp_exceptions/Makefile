SWIFT_SOURCES := main.swift
SWIFTFLAGS_EXTRAS = -L$(BUILDDIR) -I$(BUILDDIR) -I$(SRCDIR) -lWrapper
LD_EXTRAS = -Xlinker -rpath -Xlinker $(BUILDDIR)

all: libWrapper.dylib a.out

include Makefile.rules

libCppLib.dylib: CppLib/CppLib.cpp
	mkdir -p CppLib
	$(MAKE) -f $(MAKEFILE_RULES) \
		DYLIB_NAME=CppLib DYLIB_ONLY=YES CXXFLAGS_EXTRAS=-fPIC \
		DYLIB_CXX_SOURCES="CppLib/CppLib.cpp CppLib/CppLibWrapper.cpp"
	install_name_tool -id $@ $@

libWrapper.dylib: libCppLib.dylib
	$(MAKE) -f $(MAKEFILE_RULES) \
		DYLIB_NAME=Wrapper DYLIB_SWIFT_SOURCES=wrapper.swift \
		SWIFTFLAGS_EXTRAS="-I$(SRCDIR) -L. -lCppLib -module-link-name Wrapper"
