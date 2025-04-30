#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test eg00  ########


eg00: run
	

build:  $(SRC)/eg00.f
	-$(RM) eg00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/eg00.f -o eg00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) eg00.$(OBJX) check.$(OBJX) $(LIBS) -o eg00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test eg00
	eg00.$(EXESUFFIX)

verify: ;

eg00.run: run

