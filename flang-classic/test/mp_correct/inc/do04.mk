#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
do04: do04.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN4) ./a.$(EXESUFFIX) $(LOG)
do04.$(OBJX): $(SRC)/do04.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/do04.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) do04.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: do04
run: ;
