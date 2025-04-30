#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp30  ########


qp30: run


build:  $(SRC)/qp30.f08
	-$(RM) qp30.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp30.f08 -o qp30.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp30.$(OBJX) check.$(OBJX) $(LIBS) -o qp30.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp30
	qp30.$(EXESUFFIX)

verify: ;
