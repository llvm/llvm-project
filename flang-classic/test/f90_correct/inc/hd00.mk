#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test hd00  ########


hd00: run
	

build:  $(SRC)/hd00.f
	-$(RM) hd00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/hd00.f -o hd00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) hd00.$(OBJX) check.$(OBJX) $(LIBS) -o hd00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test hd00
	hd00.$(EXESUFFIX)

verify: ;

hd00.run: run

