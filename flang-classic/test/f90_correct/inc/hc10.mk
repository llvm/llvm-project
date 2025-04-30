#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test hc10  ########


hc10: run
	

build:  $(SRC)/hc10.f
	-$(RM) hc10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/hc10.f -o hc10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) hc10.$(OBJX) check.$(OBJX) $(LIBS) -o hc10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test hc10
	hc10.$(EXESUFFIX)

verify: ;

hc10.run: run

