#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test hb30  ########


hb30: run
	

build:  $(SRC)/hb30.f
	-$(RM) hb30.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/hb30.f -o hb30.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) hb30.$(OBJX) check.$(OBJX) $(LIBS) -o hb30.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test hb30
	hb30.$(EXESUFFIX)

verify: ;

hb30.run: run

