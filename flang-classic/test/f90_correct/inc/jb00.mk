#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test jb00  ########


jb00: run
	

build:  $(SRC)/jb00.f90
	-$(RM) jb00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/jb00.f90 -o jb00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) jb00.$(OBJX) check.$(OBJX) $(LIBS) -o jb00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test jb00
	jb00.$(EXESUFFIX)

verify: ;

jb00.run: run

