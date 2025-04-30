#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ea00  ########


ea00: run
	

build:  $(SRC)/ea00.f
	-$(RM) ea00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ea00.f -o ea00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ea00.$(OBJX) check.$(OBJX) $(LIBS) -o ea00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ea00
	ea00.$(EXESUFFIX)

verify: ;

ea00.run: run

