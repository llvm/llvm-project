//===-- xray_defs.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Common definitions useful for XRay sources.
//
//===----------------------------------------------------------------------===//
#ifndef XRAY_XRAY_DEFS_H
#define XRAY_XRAY_DEFS_H

#if XRAY_SUPPORTED
#define XRAY_NEVER_INSTRUMENT __attribute__((xray_never_instrument))
#else
#define XRAY_NEVER_INSTRUMENT
#endif

#endif  // XRAY_XRAY_DEFS_H
