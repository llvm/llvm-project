#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ch24  ########


ch24: run
	

build:  $(SRC)/ch24.f
	-$(RM) ch24.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ch24.f -o ch24.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ch24.$(OBJX) check.$(OBJX) $(LIBS) -o ch24.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ch24
	ch24.$(EXESUFFIX)

verify: ;

ch24.run: run

