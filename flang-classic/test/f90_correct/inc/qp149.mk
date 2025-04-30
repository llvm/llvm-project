#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp01  ########


qp01: run


build:  $(SRC)/qp01.f08
	-$(RM) qp01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp01.f08 -o qp01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp01.$(OBJX) check.$(OBJX) $(LIBS) -o qp01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp01
	qp01.$(EXESUFFIX)

verify: ;
