#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test la00  ########


la00: run
	

build:  $(SRC)/la00.f
	-$(RM) la00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/la00.f -o la00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) la00.$(OBJX) check.$(OBJX) $(LIBS) -o la00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test la00
	la00.$(EXESUFFIX)

verify: ;

la00.run: run

