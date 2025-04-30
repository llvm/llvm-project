#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test af00  ########


af00: run
	

build:  $(SRC)/af00.f
	-$(RM) af00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/af00.f -o af00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) af00.$(OBJX) check.$(OBJX) $(LIBS) -o af00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test af00
	af00.$(EXESUFFIX)

verify: ;

af00.run: run

