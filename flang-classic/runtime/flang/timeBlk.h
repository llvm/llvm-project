/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
  * \file
  * \brief tb (time block) struct (all times in seconds)
  */

/** \brief time block (all times in seconds) */
struct tb {
  double r;        /**< real elapsed time */
  double u;        /**< user time */
  double s;        /**< system time */
  double bytes;    /**< number of bytes sent */
  double datas;    /**< number of data items sent */
  double byter;    /**< number of bytes recv`d */
  double datar;    /**< number of data items recv'd */
  double bytec;    /**< number of bytes copied */
  double datac;    /**< number of data items copied */
  double maxrss;   /**< max set size */
  double minflt;   /**< minor fault */
  double majflt;   /**< major fault */
  double nsignals; /**< number of signals */
  double nvcsw;    /**< voluntary switches */
  double nivcsw;   /**< involuntary switches */
  double sbrk;     /**< sbrk value (local heap) */
  double gsbrk;    /**< sbrk value (global heap) */
  char host[256];  /**< hostname */
};
