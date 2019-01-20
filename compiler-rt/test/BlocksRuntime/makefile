#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
CCDIR=/usr/bin
#CCDIR=/Volumes/Keep/gcc/usr/bin

all: std

clean:
	rm -fr *.dSYM *.o *-bin testfilerunner

TFR = ~public/bin/testfilerunner

testfilerunner: testfilerunner.h testfilerunner.m
	gcc -fobjc-gc-only -g -arch x86_64 -arch i386 -std=gnu99 testfilerunner.m -o testfilerunner -framework Foundation

tests:
	grep CONFIG *.[cmCM] | $(TFR) $(CCDIR) --

open:
	grep CONFIG *.[cmCM] | $(TFR) $(CCDIR) -open --

fast:
	grep CONFIG *.[cmCM] | $(TFR) -fast $(CCDIR) --

std:
	grep CONFIG *.[cmCM] | $(TFR) --

clang:
	grep CONFIG *.[cmCM] | $(TFR) -clang -fast --

fastd:
	grep CONFIG *.[cmCM] | $(TFR) -fast --


# Hack Alert: arguably most of the following belongs in libclosure's Makefile; sticking it here until I get around to grokking what goes on in that file.
sudid:
	@echo Enabling sudo:  # Hack Alert: enable sudo first thing so we don't hang at the password prompt later
	@sudo echo Thanks


RootsDirectory ?= /tmp/
# Note: the libsystem project (built by the libsystemroot target below) uses the ALTUSRLOCALLIBSYSTEM variable, so we use it here to maintain parity
ALTUSRLOCALLIBSYSTEM ?= $(RootsDirectory)/alt-usr-local-lib-system/
altusrlocallibsystem:
	ditto /usr/local/lib/system $(ALTUSRLOCALLIBSYSTEM)  # FIXME: conditionalize this copy


# <rdar://problem/6456031> ER: option to not require extra privileges (-nosudo or somesuch)
Buildit ?= ~rc/bin/buildit -rootsDirectory $(RootsDirectory) -arch i386 -arch ppc -arch x86_64
blocksroot: sudid clean altusrlocallibsystem
	sudo $(Buildit) ..
	ditto $(RootsDirectory)/libclosure.roots/libclosure~dst/usr/local/lib/system $(ALTUSRLOCALLIBSYSTEM)


LibsystemVersion ?= 121
LibsystemPath ?= ~rc/Software/SnowLeopard/Projects/Libsystem/Libsystem-$(LibsystemVersion)
LibsystemTmpPath ?= $(RootsDirectory)/Libsystem-$(LibsystemVersion)
libsystemroot: blocksroot
	ditto $(LibsystemPath) $(LibsystemTmpPath)  # FIXME: conditionalize this copy
	sudo ALTUSRLOCALLIBSYSTEM=$(ALTUSRLOCALLIBSYSTEM) $(Buildit) $(LibsystemTmpPath)


# Defaults to product of the libsystemroot target but does not automatically rebuild that, make both targets if you want a fresh root
LibsystemRootPath ?= $(RootsDirectory)/Libsystem-$(LibsystemVersion).roots/Libsystem-$(LibsystemVersion)~dst/usr/lib/
roottests: 
	grep CONFIG *.[cmCM] | $(TFR) -dyld $(LibsystemRootPath) --  # FIXME: figure out if I can "call" the std target instead of duplicating it

