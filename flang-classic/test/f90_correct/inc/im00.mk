#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test im00  ########


im00: run
	

build:  $(SRC)/im00.f
	-$(RM) im00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/im00.f -o im00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) im00.$(OBJX) check.$(OBJX) $(LIBS) -o im00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test im00
	im00.$(EXESUFFIX)

verify: ;

im00.run: run

