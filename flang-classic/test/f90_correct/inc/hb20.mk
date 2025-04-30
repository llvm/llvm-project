#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test hb20  ########


hb20: run
	

build:  $(SRC)/hb20.f
	-$(RM) hb20.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/hb20.f -o hb20.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) hb20.$(OBJX) check.$(OBJX) $(LIBS) -o hb20.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test hb20
	hb20.$(EXESUFFIX)

verify: ;

hb20.run: run

