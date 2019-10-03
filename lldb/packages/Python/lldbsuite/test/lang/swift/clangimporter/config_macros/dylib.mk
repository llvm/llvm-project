LEVEL = ../../../../make
SWIFT_OBJC_INTEROP := 1
DYLIB_SWIFT_SOURCES := dylib.swift
DYLIB_NAME := Dylib

SWIFTFLAGS_EXTRAS = -Xcc -DDEBUG=1 -Xcc -D -Xcc SPACE -Xcc -U -Xcc NDEBUG

include $(LEVEL)/Makefile.rules
