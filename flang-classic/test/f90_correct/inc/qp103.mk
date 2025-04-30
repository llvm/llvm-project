#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test chartoq  ########


qp103: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp103.f08 fcheck.$(OBJX)
	-$(RM) qp103.$(EXESUFFIX) qp103.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp103.f08 -o qp103.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp103.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp103.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp103
	qp103.$(EXESUFFIX)

verify: ;

qp103.run: run

