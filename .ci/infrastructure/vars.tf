variable "cluster_name" {
  type    = string
  default = "llvm-premerge-prototype"
}

variable "externalservices_prometheus_host" {
  type    = string
  default = "https://prometheus-prod-13-prod-us-east-0.grafana.net"
}

variable "externalservices_prometheus_basicauth_username" {
  type    = number
  default = 1716097
}

variable "externalservices_loki_host" {
  type    = string
  default = "https://logs-prod-006.grafana.net"
}

variable "externalservices_loki_basicauth_username" {
  type    = number
  default = 957850
}

variable "externalservices_tempo_host" {
  type    = string
  default = "https://tempo-prod-04-prod-us-east-0.grafana.net:443"
}

variable "externalservices_tempo_basicauth_username" {
  type    = number
  default = 952165
}
