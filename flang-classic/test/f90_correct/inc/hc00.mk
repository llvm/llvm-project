#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test hc00  ########


hc00: run
	

build:  $(SRC)/hc00.f
	-$(RM) hc00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/hc00.f -o hc00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) hc00.$(OBJX) check.$(OBJX) $(LIBS) -o hc00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test hc00
	hc00.$(EXESUFFIX)

verify: ;

hc00.run: run

