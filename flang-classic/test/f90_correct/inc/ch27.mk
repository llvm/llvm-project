#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ch27  ########


ch27: run
	

build:  $(SRC)/ch27.f
	-$(RM) ch27.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ch27.f -o ch27.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ch27.$(OBJX) check.$(OBJX) $(LIBS) -o ch27.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ch27
	ch27.$(EXESUFFIX)

verify: ;

ch27.run: run

